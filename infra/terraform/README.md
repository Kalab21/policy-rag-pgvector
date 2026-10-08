# AWS reference deployment (Terraform)

> **Terraform-defined reference architecture; not currently deployed.** `terraform fmt`, `terraform validate` and a Trivy configuration scan run in CI. It needs an AWS account to apply, and the resources it creates incur AWS charges while they run.

## What it describes

The README's [end-to-end architecture](../../README.md#end-to-end-architecture) shows this deployment at a high level alongside the request path.

```
                    Internet / corporate network
                               │  (allowed CIDRs only)
                      ┌────────▼────────┐
                      │ Application     │  internal by default; a public one needs an ACM
                      │ Load Balancer   │  certificate (HTTPS, TLS 1.3 policy, HTTP redirects)
                      └────────┬────────┘
                               │ container port only
                      ┌────────▼────────┐        ┌─────────────────┐
                      │ ECS Fargate     │───────►│ CloudWatch Logs │
                      │ FastAPI service │        └─────────────────┘
                      └───┬─────────┬───┘
            PostgreSQL    │         │ HTTPS 443 only: ECR, Secrets Manager, the identity
            5432 only     │         │ provider's JWKS endpoint, optionally Bedrock
                  ┌───────▼───┐     ▼
                  │ RDS       │   AWS APIs / IdP
                  │ Postgres16│
                  │ + pgvector│  (private subnets, encrypted, not publicly accessible)
                  └───────────┘
```

| Module | Creates |
|---|---|
| `modules/network` | VPC, public and private subnets in 2-3 zones, internet gateway, optional NAT gateway, flow logs |
| `modules/ecr` | Image repository: immutable tags, scan on push, KMS encryption, lifecycle policy |
| `modules/database` | RDS PostgreSQL 16: encrypted, private, backups, IAM auth enabled, CloudWatch log exports, **master password generated and kept in Secrets Manager by RDS** (never in the configuration or in state) |
| `modules/app` | ECS cluster, Fargate task and service, load balancer, security groups, CloudWatch log group, least-privilege IAM roles |
| `environments/dev` | Wires the modules together for one environment |

## Design choices worth knowing

- **Authentication is mandatory here.** The environment sets `AUTH_MODE=jwt` and requires an issuer, an audience and a JWKS URL; the demo mode (`AUTH_MODE=off`) is not offered in the cloud. No token-signing secret exists: tokens are verified against the identity provider's public keys. This project does not provide an identity provider.
- **No secrets in the configuration.** The database user and password reach the container from Secrets Manager (`DB_USER`, `DB_PASSWORD`); the application assembles its connection URL from `DB_HOST`, `DB_PORT`, `DB_NAME` and those two values.
- **Least privilege.** The execution role can pull this one image, write this one log group and read the listed secrets. The task role has no AWS permissions unless the optional Bedrock generator is enabled, and then only `bedrock:InvokeModel` on the model ARNs you list.
- **Network.** The database accepts connections only from the application's security group. The application accepts traffic only from the load balancer. Its only outbound access is HTTPS (443) and PostgreSQL.
- **A public load balancer is opt-in and must have a certificate** (enforced by a precondition), so tokens never cross the internet over plain HTTP. `alb_ingress_cidrs` has no default.
- **Cost-conscious defaults, but this is not free.** No NAT gateway (tasks then run in public subnets behind a locked-down security group; set `enable_nat_gateway` to use private subnets), a single `db.t4g.micro` without a standby, one Fargate task. A running environment still incurs Fargate, load balancer and RDS charges; check current AWS pricing before applying, and run `terraform destroy` (after setting `deletion_protection = false`) when finished.

## Verification

- `terraform fmt -check -recursive`
- `terraform validate` on `environments/dev` (the whole module graph, including cross-module references)
- Trivy configuration scan: no HIGH or CRITICAL findings (CI fails on any). Four LOW findings remain and are accepted: log groups, ECR and Performance Insights use AWS-managed encryption rather than customer-managed KMS keys.

Applying it to an account is where ECS runtime behaviour (read-only root filesystem with a `/tmp` volume), pgvector availability in the chosen RDS region, connectivity to your identity provider and the optional Bedrock permissions are confirmed.

## Using it (outline)

```bash
cd infra/terraform/environments/dev
cp terraform.tfvars.example terraform.tfvars      # edit: image tag, allowed CIDRs, your OIDC provider
# optional: rename backend.tf.example to backend.tf for remote state
terraform init
terraform plan
terraform apply

# push the image (tags are immutable, so use a version)
aws ecr get-login-password | docker login --username AWS --password-stdin <ecr_repository_url host>
docker build -t <ecr_repository_url>:0.1.0 ../../../..
docker push <ecr_repository_url>:0.1.0
```

The application creates the `vector` extension and its tables on first start (`AUTO_INIT_SCHEMA`). Loading the sample documents is a one-off task running `python -m scripts.ingest` with the same task definition (for example `aws ecs run-task` with a command override).
