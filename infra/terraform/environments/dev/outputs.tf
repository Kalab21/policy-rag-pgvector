output "ecr_repository_url" {
  description = "Push the application image here (tag it with image_tag)."
  value       = module.ecr.repository_url
}

output "load_balancer_dns_name" {
  description = "Address of the load balancer (internal unless public_alb is set)."
  value       = module.app.alb_dns_name
}

output "ecs_cluster" {
  value = module.app.cluster_name
}

output "ecs_service" {
  value = module.app.service_name
}

output "log_group" {
  value = module.app.log_group_name
}

output "database_address" {
  value = module.database.address
}
