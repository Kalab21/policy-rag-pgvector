variable "name" {
  type = string
}

variable "vpc_id" {
  type = string
}

variable "subnet_ids" {
  description = "Private subnets for the database (at least two zones)."
  type        = list(string)
}

variable "app_security_group_id" {
  description = "Security group of the application tasks, the only source allowed to connect."
  type        = string
}

variable "engine_version" {
  description = "PostgreSQL major version. 16 is what the project is tested against."
  type        = string
  default     = "16"
}

variable "instance_class" {
  type    = string
  default = "db.t4g.micro"
}

variable "allocated_storage_gb" {
  type    = number
  default = 20
}

variable "max_allocated_storage_gb" {
  description = "Storage autoscaling ceiling."
  type        = number
  default     = 50
}

variable "database_name" {
  type    = string
  default = "policy_rag"
}

variable "master_username" {
  type    = string
  default = "policy_rag"
}

variable "multi_az" {
  description = "Second-zone standby. Roughly doubles the instance cost, so off by default."
  type        = bool
  default     = false
}

variable "backup_retention_days" {
  type    = number
  default = 7
}

variable "deletion_protection" {
  description = "Block accidental deletion. Set to false (and apply) before terraform destroy."
  type        = bool
  default     = true
}
