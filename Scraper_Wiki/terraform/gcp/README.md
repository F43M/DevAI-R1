# GCP GKE Module

This Terraform module provisions a GKE cluster for running the Scraper Wiki stack with Dask or Ray.

## Usage

```hcl
module "gke" {
  source       = "./terraform/gcp"
  project      = "my-project"
  region       = "us-central1"
  cluster_name = "scraper-wiki"
  network      = "default"
  subnetwork   = "default"
  worker_count = 4
  machine_type = "n1-standard-4"
}
```
