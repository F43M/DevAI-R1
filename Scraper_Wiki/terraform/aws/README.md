# AWS EKS Module

This Terraform module provisions an EKS cluster suitable for running the Scraper Wiki stack with Dask or Ray.

## Usage

```hcl
module "eks" {
  source        = "./terraform/aws"
  region        = "us-east-1"
  cluster_name  = "scraper-wiki"
  vpc_id        = "vpc-123456"
  subnet_ids    = ["subnet-1", "subnet-2"]
  worker_count  = 4
  instance_type = "m5.large"
}
```
