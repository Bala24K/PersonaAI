# AWS Deployment Guide (AWS Deployment-Ready)

> [!NOTE]
> This document defines the production AWS deployment architecture for PersonaAI. It is labeled as **AWS deployment-ready** to reflect that all infrastructure configurations, environment mappings, and Docker artifacts are fully prepared for AWS deployment without requiring active billing on AWS.

---

## 1. Architecture Overview

```mermaid
flowchart TD
    Client[Client / Mobile / Web] -->|HTTPS| ALB[AWS Application Load Balancer]
    
    subgraph VPC["AWS VPC (Private & Public Subnets)"]
        subgraph PublicSubnet["Public Subnets"]
            ALB
        end
        
        subgraph PrivateSubnetApp["App Subnets (ECS Fargate)"]
            API["FastAPI App Tasks (ECS)"]
            Worker["Celery Worker Tasks (ECS)"]
            Beat["Celery Beat Task (ECS)"]
        end
        
        subgraph PrivateSubnetData["Data Subnets"]
            RDS[("Amazon RDS PostgreSQL\n(Structured User/Turn State)")]
            DocDB[("Amazon DocumentDB\n(Mongo-Compatible Event Traces)")]
            ElastiCache[("Amazon ElastiCache Redis\n(Celery Results & Caching)")]
            AmazonMQ[("Amazon MQ RabbitMQ\n(Task Broker)")]
        end
    end
    
    ALB --> API
    API -->|Sync SQL Queries| RDS
    API -->|Async Tasks| AmazonMQ
    API -->|Log Traces| DocDB
    
    AmazonMQ --> Worker
    Worker -->|Execute Jobs| DocDB
    Worker -->|Store Results| ElastiCache
    Worker -->|Persist Memories| RDS
    Beat -->|Schedule Jobs| AmazonMQ
```

---

## 2. Infrastructure Service Mapping

| Component | Local Technology | AWS Production Equivalent | Responsibility |
|---|---|---|---|
| **API Layer** | FastAPI (uvicorn) | AWS ECS Fargate / App Runner | Stateless HTTP request orchestration |
| **Worker Execution** | Celery Worker | AWS ECS Fargate Tasks | Asynchronous task processing (memories, eval) |
| **Message Broker** | RabbitMQ | Amazon MQ for RabbitMQ | Decoupled message queueing & job dispatching |
| **Result Backend / Cache**| Redis | Amazon ElastiCache for Redis | Celery task state & short-lived cache |
| **Structured State** | PostgreSQL | Amazon RDS for PostgreSQL | Relational user profiles, turns, candidates |
| **Document Traces** | MongoDB | Amazon DocumentDB | Unstructured interaction logs & eval records |
| **Native Component** | C++17 Analyzer | Built into Docker Image | Subprocess binary compiled inside Linux container |

---

## 3. Production Environment Variable Configuration

Below is an example `.env.production` mapping for AWS:

```bash
# LLM Credentials (Inject via AWS Secrets Manager)
ANTHROPIC_API_KEY=arn:aws:secretsmanager:us-east-1:123456789012:secret:PersonaAI-Keys
ANTHROPIC_MODEL=claude-sonnet-4-6

# Relational Database (Amazon RDS PostgreSQL)
DATABASE_URL=postgresql+psycopg2://persona_admin:ComplexPass123!@persona-db.cluster-xyz.us-east-1.rds.amazonaws.com:5432/persona

# Message Broker (Amazon MQ RabbitMQ)
RABBITMQ_URL=amqps://b-12345678-group.mq.us-east-1.amazonaws.com:5671
CELERY_BROKER_URL=amqps://b-12345678-group.mq.us-east-1.amazonaws.com:5671

# Cache & Task Result Store (Amazon ElastiCache Redis)
REDIS_URL=rediss://persona-cache.abcxyz.clustercfg.use1.cache.amazonaws.com:6379/0
CELERY_RESULT_BACKEND=rediss://persona-cache.abcxyz.clustercfg.use1.cache.amazonaws.com:6379/0

# Document Store (Amazon DocumentDB)
MONGODB_URL=mongodb://docdbadmin:DocDBPass123!@persona-docdb.cluster-xyz.us-east-1.docdb.amazonaws.com:27017/?tls=true&tlsCAFile=rds-combined-ca-bundle.pem
MONGODB_DB_NAME=persona_ai
```

---

## 4. Container Build & Deployment Steps

1. **Build Docker Image locally or in CI/CD (GitHub Actions / AWS CodeBuild):**
   ```bash
   docker build -t persona-ai:latest .
   ```

2. **Authenticate with AWS ECR (Elastic Container Registry):**
   ```bash
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com
   ```

3. **Tag & Push Image:**
   ```bash
   docker tag persona-ai:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/persona-ai:v2.0.0
   docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/persona-ai:v2.0.0
   ```

4. **Deploy Task Definitions to ECS Fargate:**
   Update the ECS Task Definition to reference `persona-ai:v2.0.0` with task parameters for API (`uvicorn main:app --host 0.0.0.0 --port 8000`) and Worker (`celery -A app.tasks worker --loglevel=info`).

---

## 5. Security & Operational Best Practices

- **IAM Roles:** Use ECS Task Execution Roles for retrieving secrets from AWS Secrets Manager instead of hardcoding API keys.
- **VPC Isolation:** Run database clusters (RDS, DocumentDB, ElastiCache) and RabbitMQ inside private subnets inaccessible from the public Internet.
- **TLS/SSL Encryption:** Enforce TLS in transit for RDS (`sslmode=require`), DocumentDB (`tls=true`), and RabbitMQ (`amqps://`).
- **Container Permissions:** Container runs as non-root user `persona` (UID 1000).
