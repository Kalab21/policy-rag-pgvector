output "address" {
  value = aws_db_instance.this.address
}

output "port" {
  value = aws_db_instance.this.port
}

output "database_name" {
  value = aws_db_instance.this.db_name
}

output "security_group_id" {
  value = aws_security_group.db.id
}

# JSON secret with `username` and `password`, managed by RDS.
output "master_secret_arn" {
  value = aws_db_instance.this.master_user_secret[0].secret_arn
}
