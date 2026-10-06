variable "name" {
  type = string
}

variable "vpc_id" {
  type = string
}

variable "alb_subnet_ids" {
  description = "Subnets for the load balancer (public subnets if the load balancer is public)."
  type        = list(string)
}

variable "task_subnet_ids" {
  description = "Subnets for the application tasks."
  type        = list(string)
}

variable "assign_public_ip" {
  description = "Give tasks a public IP. Needed only when they run in public subnets without a NAT gateway."
  type        = bool
}

variable "database_security_group_id" {
  description = "Security group of the database, the only place the tasks may open PostgreSQL connections to."
  type        = string
}

variable "ecr_repository_arn" {
  type = string
}

variable "image" {
  description = "Full image reference, including an explicit tag, e.g. <repo-url>:1.0.0."
  type        = string
}

variable "container_port" {
  type    = number
  default = 8000
}

variable "cpu" {
  description = "Fargate CPU units. The embedding and reranking models need more than the smallest sizes."
  type        = number
  default     = 1024
}

variable "memory" {
  description = "Fargate memory in MiB."
  type        = number
  default     = 2048
}

variable "desired_count" {
  type    = number
  default = 1
}

variable "environment" {
  description = "Non-secret environment variables for the container."
  type        = map(string)
  default     = {}
}

variable "secrets" {
  description = "Environment variables sourced from Secrets Manager: name => secret ARN (optionally with :json-key::)."
  type        = map(string)
  default     = {}
}

variable "secret_arns" {
  description = "Secrets the execution role may read (the ARNs behind `secrets`, without any :json-key suffix)."
  type        = list(string)
  default     = []
}

variable "bedrock_model_arns" {
  description = "Bedrock models the application may invoke. Empty means no AWS permissions at all."
  type        = list(string)
  default     = []
}

variable "public" {
  description = "Internet-facing load balancer. Requires certificate_arn."
  type        = bool
  default     = false
}

variable "certificate_arn" {
  description = "ACM certificate for HTTPS. Empty means HTTP only, which is allowed on an internal load balancer."
  type        = string
  default     = ""
}

variable "alb_ingress_cidrs" {
  description = "Networks allowed to reach the load balancer. There is deliberately no default."
  type        = list(string)
}

variable "deletion_protection" {
  type    = bool
  default = true
}

variable "log_retention_days" {
  type    = number
  default = 30
}
