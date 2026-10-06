variable "name" {
  description = "Name prefix for the network resources."
  type        = string
}

variable "cidr" {
  description = "CIDR block of the VPC."
  type        = string
  default     = "10.20.0.0/16"
}

variable "az_count" {
  description = "Number of availability zones (one public and one private subnet each)."
  type        = number
  default     = 2

  validation {
    condition     = var.az_count >= 2 && var.az_count <= 3
    error_message = "az_count must be 2 or 3 (a load balancer and a database subnet group need at least two zones)."
  }
}

variable "enable_nat_gateway" {
  description = "Create a NAT gateway so application tasks can run in private subnets. Billed hourly, so off by default."
  type        = bool
  default     = false
}

variable "log_retention_days" {
  description = "Retention of the VPC flow log group."
  type        = number
  default     = 30
}
