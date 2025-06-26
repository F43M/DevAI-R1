terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project
  region  = var.region
}

module "gke" {
  source  = "terraform-google-modules/kubernetes-engine/google"
  version = "~> 30.0"

  project_id = var.project
  name       = var.cluster_name
  region     = var.region
  network    = var.network
  subnetwork = var.subnetwork

  node_pools = [
    {
      name       = "default"
      node_count = var.worker_count
      machine_type = var.machine_type
    }
  ]
}
