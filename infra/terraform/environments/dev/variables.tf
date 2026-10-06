variable "region" {
  description = "AWS region."
  type        = string
  default     = "us-east-1"
}

variable "name" {
  description = "Name prefix for every resource."
  type        = string
  default     = "policy-rag-platform-dev"
}

variable "image_tag" {
  description = "Tag of the application image in the ECR repository. Tags are immutable, so use a version, not 'latest'."
  type        = string

  validation {
    condition     = var.image_tag != "" && var.image_tag != "latest"
    error_message = "image_tag must be an explicit version tag, not empty or 'latest'."
  }
}

variable "alb_ingress_cidrs" {
  description = "Networks allowed to reach the load balancer. No default: choose deliberately."
  type        = list(string)
}

variable "public_alb" {
  description = "Internet-facing load balancer. Requires certificate_arn."
  type        = bool
  default     = false
}

variable "certificate_arn" {
  description = "ACM certificate for HTTPS."
  type        = string
  default     = ""
}

variable "enable_nat_gateway" {
  description = "Run tasks in private subnets behind a NAT gateway (billed hourly). Off: tasks use public subnets with a locked-down security group."
  type        = bool
  default     = false
}

# Authentication is required in this environment, so these have no defaults.
variable "auth_issuer" {
  description = "Expected token issuer (the identity provider's issuer URL)."
  type        = string
}

variable "auth_audience" {
  description = "Expected token audience."
  type        = string
}

variable "auth_jwks_url" {
  description = "The identity provider's JWKS endpoint, used to verify token signatures."
  type        = string
}

variable "desired_count" {
  type    = number
  default = 1
}

variable "db_instance_class" {
  type    = string
  default = "db.t4g.micro"
}

variable "deletion_protection" {
  description = "Protect the database and load balancer from deletion. Set false and apply before terraform destroy."
  type        = bool
  default     = true
}

variable "rerank_enabled" {
  description = "Enable cross-encoder reranking (more CPU and latency)."
  type        = bool
  default     = false
}

variable "bedrock_model_id" {
  description = "Enable the optional Bedrock generator with this model id. Empty keeps the extractive generator."
  type        = string
  default     = ""
}

variable "bedrock_model_arns" {
  description = "Bedrock model ARNs the application may invoke (required with bedrock_model_id)."
  type        = list(string)
  default     = []
}
