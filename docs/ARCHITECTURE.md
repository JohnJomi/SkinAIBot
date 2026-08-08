# SkinAIBot Architecture Documentation

## Project Overview

SkinAIBot is an AI-powered platform for skin disease diagnosis. The backend provides a RESTful API for image upload, AI-powered skin condition analysis, explainability reports, and a conversational chatbot for patient guidance.

**Tech Stack:**
- **Backend:** FastAPI (Python 3.11+)
- **AI/ML:** Azure Custom Vision / ONNX Runtime / PyTorch
- **Storage:** Azure Blob Storage
- **Database:** PostgreSQL (planned)
- **Deployment:** Azure Container Apps / AKS

---

## High-Level System Architecture

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
                    │  (Custom Vision) │              │
                    └──────────────────┘              │
                                                       │
                    ┌──────────────────┐              │
                    │  PostgreSQL      │◀─────────────┤
                    │  (Future)        │              │
                    └──────────────────┘              │
```

---

## Backend Architecture

### Layered Architecture

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
├── ai/               # AI Layer - ML model integration
│   ├── adapters/     # Model interface adapters
│   ├── providers/    # Provider implementations
│   └── prompts/      # Prompt templates
├── exceptions/       # Error Layer - Custom exceptions & handlers
├── logging/          # Observability Layer - Logging config
├── middleware/       # Cross-cutting Concerns - Middleware
├── utils/            # Shared Utilities
└── tests/            # Test Suite
```

### Request Flow

```
HTTP Request
    │
    ▼
Middleware (CORS, Logging, Error Handling)
    │
    ▼
API Router (v1)
    │
    ▼
Dependency Injection (Auth, DB Session, Storage Client)
    │
    ▼
Service Layer (Business Logic)
    │
    ├──▶ Repository Layer (Data Access)
    │
    ├──▶ Storage Layer (File Operations)
    │
    └──▶ AI Layer (Model Inference)
    │
    ▼
Response Serialization (Pydantic Schemas)
    │
    ▼
HTTP Response
```

---

## Planned Frontend Architecture

- **Framework:** React 18+ with TypeScript
- **State Management:** Zustand / React Query
- **UI Library:** Tailwind CSS + shadcn/ui
- **Build Tool:** Vite
- **Deployment:** Azure Static Web Apps / Vercel

### Frontend-Backend Contract

- OpenAPI 3.0 specification auto-generated from FastAPI
- TypeScript types generated via `openapi-typescript`
- API client generated via `orval` or `openapi-fetch`

---

## AI Architecture

### Components

| Component | Responsibility | Technology |
|-----------|---------------|------------|
| Image Classifier | Skin disease classification | Azure Custom Vision / PyTorch |
| Explainability | Grad-CAM / LIME heatmaps | Captum / Custom |
| Chatbot | Patient Q&A | Azure OpenAI / Llama |
| Prompt Templates | Structured prompts | Jinja2 / LangChain |

### Inference Pipeline

```
Uploaded Image
     │
     ▼
Preprocessing (resize, normalize)
     │
     ▼
Model Inference (Classification)
     │
     ├──▶ Top-K Predictions + Confidence
     │
     ▼
Explainability (Grad-CAM)
     │
     ▼
Structured Response
```

### Provider Abstraction

```python
# ai/providers/base.py
class AIProvider(ABC):
    @abstractmethod
    async def predict(self, image: bytes) -> PredictionResult: ...

    @abstractmethod
    async def explain(self, image: bytes, class_idx: int) -> ExplanationResult: ...
```

Providers: `AzureCustomVisionProvider`, `LocalONNXProvider`, `PyTorchProvider`

---

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

---

## Database Architecture (Planned)

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

---

## Folder Structure and Responsibilities

| Folder | Responsibility | Public API |
|--------|---------------|------------|
| `api/v1/routers/` | HTTP endpoints, request validation | Router objects |
| `api/dependencies/` | DI providers (DB, Auth, Storage) | Dependency functions |
| `core/` | App factory, lifespan, base classes | `create_app()` |
| `config/` | Settings via Pydantic BaseSettings | `settings` singleton |
| `database/` | Engine, session, base model | `get_session()`, `Base` |
| `models/` | SQLAlchemy models | Model classes |
| `schemas/` | Pydantic request/response models | Schema classes |
| `repositories/` | DB queries, CRUD operations | Repository classes |
| `services/` | Business logic, orchestration | Service classes |
| `storage/` | File upload/download/presigned URLs | Storage backend |
| `ai/adapters/` | Unified model interfaces | Adapter classes |
| `ai/providers/` | Provider implementations | Provider classes |
| `ai/prompts/` | Prompt templates | Template functions |
| `exceptions/` | Custom exceptions, handlers | Exception classes |
| `logging/` | Structured logging config | `get_logger()` |
| `middleware/` | Request/response middleware | Middleware classes |
| `utils/` | Pure helper functions | Functions |
| `tests/` | Unit, integration, e2e tests | Test modules |

---

## Layer Responsibilities

| Layer | Responsibilities | Must Not |
|-------|-----------------|----------|
| **API** | Routing, validation, serialization, HTTP concerns | Business logic, DB queries |
| **Services** | Use cases, transactions, orchestration | HTTP handling, raw SQL |
| **Repositories** | Data access, queries, mapping | Business logic, HTTP |
| **Models** | Database schema, relationships | Business logic |
| **Schemas** | Validation, serialization, API contracts | DB operations |
| **AI** | Model inference, preprocessing, postprocessing | HTTP, DB |
| **Storage** | File operations, signed URLs | Business logic |

---

## Dependency Rules

```
api ──▶ services ──▶ repositories ──▶ models
  │         │              │
  │         ├──▶ storage
  │         ├──▶ ai
  │         └──▶ schemas
  │
  └──▶ dependencies (DI)
```

**Rules:**
1. Upper layers depend on lower layers (not reverse)
2. Services orchestrate; Repositories isolate DB
3. Schemas are shared contracts (no DB models in API responses)
4. AI/Storage accessed only via Services
5. Config accessed only via `config.settings`

---

## Coding Conventions

### Python Style

- **Formatter:** Ruff (line length: 100)
- **Type Hints:** Required for all public functions
- **Imports:** `isort` (std → third-party → local)
- **Async:** Default to `async`/`await` for I/O

### Code Organization

```python
# Module docstring
"""Brief description of module purpose."""

# Standard library imports
import asyncio
from datetime import UTC, datetime
from uuid import UUID

# Third-party imports
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

# Local imports
from app.config import settings
from app.database import get_session
from app.schemas.analysis import AnalysisCreate, AnalysisResponse

# Constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Classes/Functions
class AnalysisService:
    """Service for analysis operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, data: AnalysisCreate) -> AnalysisResponse:
        ...
```

---

## Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Modules | `snake_case` | `analysis_service.py` |
| Classes | `PascalCase` | `AnalysisService` |
| Functions/Variables | `snake_case` | `create_analysis()` |
| Constants | `UPPER_SNAKE_CASE` | `MAX_FILE_SIZE` |
| Pydantic Models | `PascalCase` + Suffix | `AnalysisCreate`, `AnalysisResponse` |
| SQLAlchemy Models | `PascalCase` | `Analysis` |
| Database Tables | `snake_case`, plural | `analyses` |
| Enum Values | `UPPER_SNAKE_CASE` | `Status.COMPLETED` |
| Environment Variables | `UPPER_SNAKE_CASE` | `DATABASE_URL` |

---

## Error Handling Philosophy

### Principles

1. **Fail Fast** - Validate early, return 4xx for client errors
2. **Structured Errors** - Consistent error response format
3. **No Stack Traces** - Never leak internals to clients
4. **Actionable Messages** - Tell users what to do
5. **Correlation IDs** - Trace requests across services

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": [
      {"field": "image", "issue": "File too large (max 10MB)"}
    ],
    "request_id": "req_abc123"
  }
}
```

### Exception Hierarchy

```
AppException (base)
├── ValidationError (400)
├── AuthenticationError (401)
├── AuthorizationError (403)
├── NotFoundError (404)
├── ConflictError (409)
├── RateLimitError (429)
├── StorageError (500)
├── AIModelError (500)
└── InternalError (500)
```

---

## Logging Philosophy

### Structured Logging (JSON)

```python
logger.info(
    "analysis_created",
    extra={
        "analysis_id": str(analysis.id),
        "user_id": str(user.id),
        "image_size": len(image_bytes),
        "duration_ms": 145
    }
)
```

### Log Levels

| Level | Use Case |
|-------|----------|
| DEBUG | Detailed diagnostic info |
| INFO | Request/response, business events |
| WARNING | Recoverable issues, retries |
| ERROR | Failed operations, exceptions |
| CRITICAL | System-level failures |

### Fields (Always Include)

- `timestamp` (ISO 8601, UTC)
- `level`
- `logger` (module name)
- `message`
- `request_id` (from middleware)
- `user_id` (when authenticated)

---

## API Design Principles

### RESTful Conventions

| Resource | Endpoints |
|----------|-----------|
| Analyses | `POST /api/v1/analyses` (create)<br>`GET /api/v1/analyses/{id}`<br>`GET /api/v1/analyses` (list)<br>`DELETE /api/v1/analyses/{id}` |
| Chat | `POST /api/v1/chat/sessions`<br>`POST /api/v1/chat/sessions/{id}/messages`<br>`GET /api/v1/chat/sessions/{id}/messages` |
| Health | `GET /health` |

### Versioning

- URL versioning: `/api/v1/`
- Breaking changes → new version
- Deprecation header: `Deprecation: true`

### Response Envelope

```json
{
  "data": {},
  "meta": {
    "page": 1,
    "page_size": 20,
    "total": 100
  }
}
```

---

## Security Principles

### Authentication (Planned)

- JWT tokens (RS256)
- Short-lived access tokens (15min)
- Refresh tokens (7 days, rotating)
- Token blacklist on logout

### Authorization

- Role-based: `user`, `admin`
- Resource ownership checks in Services

### Data Protection

- HTTPS only (enforced by Azure)
- PII encryption at rest
- No secrets in code (Azure Key Vault)
- Input validation on all endpoints

### File Upload Security

- MIME type validation
- File size limits (10MB)
- Virus scanning (planned)
- Signed URLs for direct upload

---

## Future Scalability Considerations

### Horizontal Scaling

- Stateless API pods (FastAPI + Uvicorn workers)
- Redis for session/cache (planned)
- Database read replicas
- CDN for static assets

### Async Processing

- Background jobs via Celery + Redis (planned)
- AI inference queue for high load
- Webhook callbacks for long-running tasks

### Observability

- OpenTelemetry tracing
- Prometheus metrics (`/metrics`)
- Structured logs → Azure Monitor / ELK
- Health checks: liveness, readiness

### Multi-Region

- Active-passive for DR
- Geo-distributed storage
- Latency-based routing

---

## Appendix: Current Implementation Status

| Component | Status |
|-----------|--------|
| FastAPI App | ✅ Initialized |
| Folder Structure | ✅ Complete |
| Health Endpoint | ✅ Implemented |
| Configuration | 🔄 In Progress |
| Database | ⏳ Planned |
| Authentication | ⏳ Planned |
| Image Upload | ⏳ Planned |
| AI Pipeline | ⏳ Planned |
| Chatbot | ⏳ Planned |
| Frontend | ⏳ Planned |
| Deployment | ⏳ Planned |