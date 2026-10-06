terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Remote state is recommended for anything that is actually applied. See backend.tf.example.
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project     = "policy-rag-platform"
      Environment = "dev"
      ManagedBy   = "terraform"
    }
  }
}
