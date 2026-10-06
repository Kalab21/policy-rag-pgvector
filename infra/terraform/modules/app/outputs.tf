output "alb_dns_name" {
  value = aws_lb.this.dns_name
}

output "app_security_group_id" {
  value = aws_security_group.app.id
}

output "cluster_name" {
  value = aws_ecs_cluster.this.name
}

output "service_name" {
  value = aws_ecs_service.app.name
}

output "log_group_name" {
  value = aws_cloudwatch_log_group.app.name
}
