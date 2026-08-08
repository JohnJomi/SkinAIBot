# Architecture

The SkinAIBot platform follows a Two-Team Architecture with distinct separation of concerns between the Application (Full Stack) team and the AI/ML Platform team.

## Overview

```mermaid
flowchart TD
    subgraph "Application (Team B)"
    F[Frontend (React/Vite)] --> B[Backend (FastAPI)]
    B --> DB[(PostgreSQL)]
    B --> Storage[(Blob Storage)]
    end

    subgraph "AI Platform (Team A)"
    B -- "REST API Contract" --> AI[AI Mock Service]
    AI --> M[ML Pipeline]
    M -. "Future" .-> EN[EfficientNet]
    M -. "Future" .-> LLM[GPT/Claude]
    end
```

## Team A: AI/ML Platform

Owns all machine learning capabilities:
- Dataset preparation
- Preprocessing and augmentation
- Model architecture (EfficientNet)
- Training and MLOps
- Inference and prediction pipelines
- Confidence logic and fallback mechanisms
- LLM integrations and prompt engineering
- Explainability and AI safety evaluation
- The AI Service API

## Team B: Application (Full Stack)

Owns the user product and infrastructure:
- Frontend user interface (React)
- Authentication and User Management
- The FastAPI Gateway
- File uploads and Storage (Blob Storage)
- Database (PostgreSQL)
- Chat histories, reports, and audit logs
- Calling the AI Service via the versioned API contract
- Application deployment

## API Contract

The teams are decoupled via an explicit REST API contract located in `contracts/ai-api/`. 
Neither team should depend on the other's internal implementation details.

---

# Detailed Application Backend Architecture

## High-Level System Architecture (Backend)

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Frontend   │────▶│   API Gateway    │────▶│  FastAPI App    │
│  (React)    │     │   (Azure/API GW) │     │                 │
└─────────────┘     └──────────────────┘     └────────┬────────┘
                                                       │
                    ┌──────────────────┐              │
                    │  Azure Blob      │◀─────────────┤
                    │  Storage         │              │
                    └──────────────────┘              │
                                                       │
                    ┌──────────────────┐              │
                    │  AI Services     │◀─────────────┤
                    │  (REST API)      │              │
                    └──────────────────┘              │
                                                       │
                    ┌──────────────────┐              │
                    │  PostgreSQL      │◀─────────────┤
                    │                  │              │
                    └──────────────────┘              │
```

## Layered Architecture

```
app/
├── api/              # API Layer - HTTP handling, routing, validation
│   ├── v1/           # API Version 1
│   └── dependencies/ # FastAPI dependency injection
├── core/             # Core Layer - App initialization, config, base classes
├── config/           # Configuration Layer - Settings management
├── database/         # Data Layer - DB connections, sessions, migrations
├── models/           # Domain Layer - Database models (SQLAlchemy)
├── schemas/          # Contract Layer - Pydantic request/response models
├── repositories/     # Data Access Layer - Database operations
├── services/         # Business Logic Layer - Use cases, orchestration
├── storage/          # Storage Layer - File storage abstractions
├── ai/               # AI Layer - AI Service Client
│   ├── client.py     # AI Service Client
│   └── schemas.py    # AI Service Contract
├── exceptions/       # Error Layer - Custom exceptions & handlers
├── logging/          # Observability Layer - Logging config
├── middleware/       # Cross-cutting Concerns - Middleware
├── utils/            # Shared Utilities
└── tests/            # Test Suite
```

## Storage Architecture

### Azure Blob Storage Structure

```
container: skin-ai-images/
├── uploads/
│   └── {user_id}/{uuid}.jpg
├── processed/
│   └── {user_id}/{uuid}_heatmap.jpg
└── models/
    └── {model_version}.onnx
```

### Storage Abstraction

```python
# storage/base.py
class StorageBackend(ABC):
    @abstractmethod
    async def upload(self, path: str, data: bytes, content_type: str) -> str: ...

    @abstractmethod
    async def download(self, path: str) -> bytes: ...

    @abstractmethod
    async def delete(self, path: str) -> None: ...

    @abstractmethod
    async def generate_presigned_url(self, path: str, expiry: int) -> str: ...
```

Implementations: `AzureBlobStorage`, `LocalStorage` (dev)

## Database Architecture

### PostgreSQL Schema Overview

```sql
-- Users & Authentication
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Analysis Records
CREATE TABLE analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    image_path VARCHAR(500) NOT NULL,
    heatmap_path VARCHAR(500),
    predictions JSONB NOT NULL,
    top_prediction VARCHAR(100),
    confidence DECIMAL(5,4),
    status VARCHAR(20) DEFAULT 'completed',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Chat Sessions
CREATE TABLE chat_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    analysis_id UUID REFERENCES analyses(id),
    title VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES chat_sessions(id),
    role VARCHAR(20) NOT NULL, -- 'user' | 'assistant' | 'system'
    content TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### ORM: SQLAlchemy 2.0 (Async)

- Async session management in `database/session.py`
- Repository pattern in `repositories/`
- Migrations via Alembic
