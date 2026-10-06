locals {
  bedrock_enabled = var.bedrock_model_id != ""

  environment = merge(
    {
      # The database is reached by parts; the password and user come from Secrets Manager.
      DB_HOST    = module.database.address
      DB_PORT    = tostring(module.database.port)
      DB_NAME    = module.database.database_name
      DB_SSLMODE = "require"

      # Authentication is mandatory in the cloud: the demo mode (AUTH_MODE=off) is not offered.
      AUTH_MODE     = "jwt"
      AUTH_ISSUER   = var.auth_issuer
      AUTH_AUDIENCE = var.auth_audience
      AUTH_JWKS_URL = var.auth_jwks_url

      RERANK_ENABLED = tostring(var.rerank_enabled)
      LOG_FORMAT     = "json"
    },
    local.bedrock_enabled ? {
      LLM_PROVIDER     = "bedrock"
      BEDROCK_MODEL_ID = var.bedrock_model_id
      BEDROCK_REGION   = var.region
    } : {}
  )
}

# Cross-variable checks that a variable validation cannot express.
resource "terraform_data" "guards" {
  lifecycle {
    precondition {
      condition     = !local.bedrock_enabled || length(var.bedrock_model_arns) > 0
      error_message = "bedrock_model_id needs bedrock_model_arns: the application role may only invoke models you list."
    }
    precondition {
      condition     = !var.public_alb || var.certificate_arn != ""
      error_message = "public_alb needs certificate_arn: tokens must not cross the internet over plain HTTP."
    }
  }
}

module "network" {
  source = "../../modules/network"

  name               = var.name
  enable_nat_gateway = var.enable_nat_gateway
}

module "ecr" {
  source = "../../modules/ecr"

  name = var.name
}

module "database" {
  source = "../../modules/database"

  name                  = var.name
  vpc_id                = module.network.vpc_id
  subnet_ids            = module.network.private_subnet_ids
  app_security_group_id = module.app.app_security_group_id
  instance_class        = var.db_instance_class
  deletion_protection   = var.deletion_protection
}

module "app" {
  source = "../../modules/app"

  name                       = var.name
  vpc_id                     = module.network.vpc_id
  alb_subnet_ids             = var.public_alb ? module.network.public_subnet_ids : module.network.private_subnet_ids
  task_subnet_ids            = module.network.app_subnet_ids
  assign_public_ip           = module.network.app_assign_public_ip
  database_security_group_id = module.database.security_group_id
  ecr_repository_arn         = module.ecr.repository_arn

  image         = "${module.ecr.repository_url}:${var.image_tag}"
  desired_count = var.desired_count
  environment   = local.environment

  secrets = {
    DB_USER     = "${module.database.master_secret_arn}:username::"
    DB_PASSWORD = "${module.database.master_secret_arn}:password::"
  }
  secret_arns = [module.database.master_secret_arn]

  bedrock_model_arns = var.bedrock_model_arns

  public              = var.public_alb
  certificate_arn     = var.certificate_arn
  alb_ingress_cidrs   = var.alb_ingress_cidrs
  deletion_protection = var.deletion_protection
}
