# Deployment Guide

## Overview

This guide covers deploying RetailSync AI to various environments, from local development to cloud production.

## Local Deployment

### Option 1: Direct Python

```bash
# Clone repository
git clone https://github.com/<your-username>/retailsync-ai.git
cd retailsync-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run pipeline (if not already done)
python src/data/generate_dataset.py
python src/data/ingest.py
python src/database/init_db.py
python src/features/feature_engineering.py
python src/forecasting/demand_forecaster.py
python src/forecasting/forecast_pipeline.py
python src/inventory/inventory_intelligence.py
python src/anomaly/anomaly_detection.py
python src/clustering/segmentation.py

# Run dashboard
streamlit run dashboard/app.py
```

### Option 2: Docker (Local)

```bash
# Build image
docker build -t retailsync-ai .

# Run container
docker run -p 8501:8501 \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/database:/app/database \
    retailsync-ai
```

### Option 3: Docker Compose (Local)

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Access dashboard at: `http://localhost:8501`

## Cloud Deployment

### AWS Deployment

#### Prerequisites
- AWS account
- Docker installed locally
- AWS CLI configured

#### Steps

1. **Build and push Docker image:**

```bash
# Build image
docker build -t retailsync-ai .

# Tag for ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker tag retailsync-ai:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/retailsync-ai:latest

# Push to ECR
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/retailsync-ai:latest
```

2. **Deploy to ECS:**

```bash
# Create ECS task definition
# Update task-definition.json with your ECR image URL

# Register task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create/update service
aws ecs create-service --cluster retailsync-cluster --service-name retailsync-ai --task-definition retailsync-ai --desired-count 1 --launch-type FARGATE --assign-public-ip ENABLED
```

3. **Configure Application Load Balancer:**

```bash
# Create target group
aws elbv2 create-target-group --name retailsync-targets --protocol HTTP --port 8501 --vpc-id <vpc-id> --target-type ip

# Create listener
aws elbv2 create-listener --load-balancer-arn <lb-arn> --protocol HTTP --port 80 --default-actions Type=forward,TargetGroupArn=<target-group-arn>
```

### GCP Deployment

#### Option 1: Cloud Run

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/<project-id>/retailsync-ai

# Deploy to Cloud Run
gcloud run deploy retailsync-ai \
    --image gcr.io/<project-id>/retailsync-ai \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --port 8501 \
    --memory 2Gi \
    --cpu 2
```

#### Option 2: GKE

```bash
# Create cluster
gcloud container clusters create retailsync-cluster --zone us-central1-a --num-nodes 2

# Deploy using kubectl
kubectl apply -f k8s-deployment.yaml
kubectl apply -f k8s-service.yaml
```

### Azure Deployment

#### Option 1: Container Instances

```bash
# Build and push to ACR
az acr build --registry <registry-name> --image retailsync-ai .

# Deploy to Container Instances
az container create \
    --resource-group <resource-group> \
    --name retailsync-ai \
    --image <registry-name>.azurecr.io/retailsync-ai \
    --ports 8501 \
    --cpu 2 \
    --memory 4 \
    --environment-variables STREAMLIT_SERVER_PORT=8501
```

#### Option 2: AKS

```bash
# Create AKS cluster
az aks create --resource-group <resource-group> --name retailsync-cluster --node-count 2 --enable-addons monitoring

# Deploy using kubectl
kubectl apply -f k8s-deployment.yaml
kubectl apply -f k8s-service.yaml
```

## Kubernetes Deployment

### Deployment YAML

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: retailsync-ai
  labels:
    app: retailsync-ai
spec:
  replicas: 2
  selector:
    matchLabels:
      app: retailsync-ai
  template:
    metadata:
      labels:
        app: retailsync-ai
    spec:
      containers:
      - name: retailsync-ai
        image: retailsync-ai:latest
        ports:
        - containerPort: 8501
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /
            port: 8501
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /
            port: 8501
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: retailsync-ai
spec:
  selector:
    app: retailsync-ai
  ports:
  - port: 80
    targetPort: 8501
  type: LoadBalancer
```

Deploy:

```bash
kubectl apply -f k8s-deployment.yaml
```

## Environment Variables

Create `.env` file in project root:

```env
# Database
DATABASE_URL=sqlite:///database/retailsync.db

# Application
APP_ENV=production
DEBUG=false

# Streamlit
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

## Production Checklist

- [ ] Set `DEBUG=false`
- [ ] Use production database (PostgreSQL)
- [ ] Enable authentication
- [ ] Configure logging
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Configure backups
- [ ] Set up CI/CD pipeline
- [ ] Enable HTTPS with SSL certificates
- [ ] Configure firewall rules
- [ ] Set resource limits (CPU, memory)
- [ ] Implement health checks
- [ ] Set up alerting

## Monitoring

### Health Checks

```bash
# Check if dashboard is responding
curl http://localhost:8501/_stcore/health

# Check database connection
python -c "from sqlalchemy import create_engine; create_engine('sqlite:///database/retailsync.db').connect()"
```

### Metrics to Monitor

- Dashboard response time
- Database query performance
- Memory usage
- CPU usage
- Error rates
- User sessions

## Backup and Recovery

### Database Backup

```bash
# SQLite backup
cp database/retailsync.db database/retailsync.db.backup

# Automated daily backup
0 2 * * * cp database/retailsync.db database/backups/retailsync-$(date +\%Y\%m\%d).db
```

### Model Backup

```bash
# Backup models
tar -czf models-backup-$(date +\%Y\%m\%d).tar.gz models/
```

## Scaling

### Horizontal Scaling

- Deploy multiple dashboard instances behind a load balancer
- Use shared database (PostgreSQL)
- Implement session affinity if needed

### Vertical Scaling

- Increase CPU/memory allocations
- Use larger instance types
- Optimize data loading (cache more aggressively)

## Troubleshooting

### Container won't start

```bash
# Check logs
docker logs retailsync-ai

# Common issues:
# - Port 8501 already in use
# - Insufficient memory
# - Missing volume mounts
```

### Database connection issues

```bash
# Verify database file exists
ls -la database/retailsync.db

# Check file permissions
chmod 644 database/retailsync.db
```

### Performance issues

```bash
# Increase Streamlit cache TTL
# Reduce dataset size for development
# Use database indexes for faster queries
```

## Security Considerations

1. **Network:** Use firewall rules to restrict access
2. **Authentication:** Implement Streamlit authentication or reverse proxy auth
3. **Secrets:** Store in environment variables, not in code
4. **Updates:** Regularly update base images and dependencies
5. **Scanning:** Scan images for vulnerabilities (`docker scan`)

## Cost Optimization

1. **Right-size instances:** Match CPU/memory to actual usage
2. **Use spot instances:** For non-critical workloads
3. **Enable auto-scaling:** Scale down during low traffic
4. **Cache aggressively:** Reduce database load
5. **Compress data:** Reduce storage and transfer costs
