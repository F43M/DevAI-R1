# Scaling Guidelines

This document provides high level recommendations for deploying Scraper Wiki in a distributed environment.

## Worker Sizing

- **API**: Typically 1-2 replicas are sufficient. Each instance requires about **0.5 CPU** and **512Mi** of memory.
- **Scraper Workers**: Start with the number of replicas defined in `cluster.yaml` (`workers: 4`). Each worker consumes roughly **1 CPU** and **1Gi** of memory.
- **Dask/Ray Workers**: Match the value in `cluster.yaml`. A baseline of **1 CPU** and **2Gi** per worker is recommended.

## Scheduler and Queue

- The Dask scheduler is lightweight and can run with **0.5 CPU** and **512Mi** memory.
- RabbitMQ should run with persistent storage (5–10Gi) and at least **0.5 CPU**.

## Storage

- Prometheus requires persistent volume of 10–20Gi for metrics retention.
- Application logs and datasets can be stored on network volumes or object storage such as S3 or GCS.

## Autoscaling

Horizontal Pod Autoscaling (HPA) can be enabled on the worker deployments using CPU utilization targets. An example configuration is provided in `k8s/hpa-worker.yaml` which scales the `scraper-worker` Deployment between one and ten replicas when average CPU usage exceeds **75%**. Adjust the `averageUtilization` threshold and `maxReplicas` values to match your cluster capacity and workload characteristics.

HPA requires the Kubernetes **metrics server** to be installed so the cluster can collect resource metrics. Ensure your cluster has it running (e.g. `helm install metrics-server bitnami/metrics-server`) before applying the autoscaler. Increase the node count in your Terraform modules if additional capacity is needed.

## Metrics Scraping

All components expose Prometheus metrics on port **8001**. To scrape these
metrics configure Prometheus as follows:

```yaml
scrape_configs:
  - job_name: 'scraper-wiki'
    static_configs:
      - targets: ['<service-host>:8001']
    metrics_path: /
```

## Worker Restarts

Scraper workers are stateless and can be restarted at any time. When deploying
with **systemd**, define a service similar to:

```ini
[Service]
ExecStart=/usr/bin/python /opt/scraper-wiki/worker.py
Restart=always
```

On **Kubernetes**, use a Deployment with `restartPolicy: Always` and scale the
`worker` pod replicas as needed:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: scraper-worker
spec:
  replicas: 4
  template:
    spec:
      restartPolicy: Always
```
