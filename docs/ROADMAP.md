# SkinAIBot Project Roadmap

## Vision

Build a production-ready, scalable AI platform for skin disease diagnosis that empowers users with accurate, explainable AI predictions and intelligent conversational guidance.

**Target Users:** Patients seeking preliminary skin condition assessment, dermatologists for triage assistance, healthcare providers for telemedicine integration.

**Core Value Proposition:** Fast, accessible, explainable skin analysis with AI-powered chatbot support.

---

## Current Progress

| Area | Status | Notes |
|------|--------|-------|
| Project Setup | ✅ Complete | FastAPI initialized, folder structure created |
| Architecture Docs | ✅ Complete | ARCHITECTURE.md created |
| Configuration | 🔄 In Progress | Pydantic Settings needed |
| Database | ⏳ Not Started | PostgreSQL + Alembic |
| Authentication | ⏳ Not Started | JWT + Refresh tokens |
| Image Upload | ⏳ Not Started | Upload + validation endpoints |
| Azure Storage | ⏳ Not Started | Blob Storage abstraction + presigned URLs |
| AI Prediction Pipeline | ⏳ Not Started | Azure Custom Vision / ONNX |
| Explainability | ⏳ Not Started | Grad-CAM heatmaps |
| Chatbot | ⏳ Not Started | Azure OpenAI |
| Frontend | ⏳ Not Started | React + TypeScript |
| Deployment | ⏳ Not Started | Azure Container Apps |
| Monitoring | ⏳ Not Started | OpenTelemetry + Prometheus |

---

## Sprint Roadmap

### Phase 1: Foundation (Sprints 1-4)

| Sprint | Objective | Status | Target |
|--------|-----------|--------|--------|
| **Sprint 1** | Backend Foundation | 🟢 Ready to Start | Week 1-2 |
| **Sprint 2** | Authentication | ⏳ Pending | Week 3-4 |
| **Sprint 3** | Image Upload | ⏳ Pending | Week 5-6 |
| **Sprint 4** | Azure Storage | ⏳ Pending | Week 7-8 |

### Phase 2: AI Core (Sprints 5-6)

| Sprint | Objective | Status | Target |
|--------|-----------|--------|--------|
| **Sprint 5** | AI Prediction Pipeline | ⏳ Pending | Week 9-10 |
| **Sprint 6** | Explainability | ⏳ Pending | Week 11-12 |

### Phase 3: Intelligence & Interface (Sprints 7-8)

| Sprint | Objective | Status | Target |
|--------|-----------|--------|--------|
| **Sprint 7** | Chatbot | ⏳ Pending | Week 13-14 |
| **Sprint 8** | Frontend | ⏳ Pending | Week 15-16 |

### Phase 4: Production (Sprints 9-10)

| Sprint | Objective | Status | Target |
|--------|-----------|--------|--------|
| **Sprint 9** | Deployment | ⏳ Pending | Week 17-18 |
| **Sprint 10** | Monitoring | ⏳ Pending | Week 19-20 |

---

## Sprint Details

### Sprint 1 – Backend Foundation

**Goal:** Establish solid backend infrastructure

| Task | Status |
|------|--------|
| Pydantic Settings configuration | ☐ |
| Database connection (SQLAlchemy async) | ☐ |
| Alembic migration setup | ☐ |
| Base models & repositories | ☐ |
| Exception hierarchy & handlers | ☐ |
| Structured logging (JSON) | ☐ |
| Request ID middleware | ☐ |
| Health check enhancements | ☐ |
| Unit test infrastructure (pytest) | ☐ |
| Pre-commit hooks (ruff, mypy) | ☐ |

**Definition of Done:** App starts, connects to DB, runs migrations, logs structured JSON, passes lint/type checks.

---

### Sprint 2 – Authentication

**Goal:** Secure user authentication and authorization

| Task | Status |
|------|--------|
| User model & schema | ☐ |
| Password hashing (bcrypt/argon2) | ☐ |
| JWT access tokens (RS256) | ☐ |
| Refresh token rotation | ☐ |
| Login / Register endpoints | ☐ |
| Password reset flow | ☐ |
| Token blacklist (Redis) | ☐ |
| Auth dependencies (get_current_user) | ☐ |
| Role-based access (user/admin) | ☐ |
| Integration tests | ☐ |

**Definition of Done:** Users can register, login, access protected endpoints, tokens rotate securely.

---

### Sprint 3 – Image Upload

**Goal:** Secure, validated image ingestion

| Task | Status |
|------|--------|
| Upload endpoint (multipart) | ☐ |
| File validation (MIME, size, magic bytes) | ☐ |
| Presigned URL upload flow | ☐ |
| Direct upload endpoint | ☐ |
| Upload progress tracking | ☐ |
| Image preprocessing pipeline (resize, normalize) | ☐ |
| Analysis record creation on upload | ☐ |
| Integration tests | ☐ |

**Definition of Done:** Users can upload validated images and trigger an analysis record.

---

### Sprint 4 – Azure Storage

**Goal:** Production-grade Azure Blob Storage integration

| Task | Status |
|------|--------|
| Azure Blob Storage client | ☐ |
| Storage abstraction interface | ☐ |
| Presigned URL generation (PUT/GET) | ☐ |
| Container & lifecycle policies | ☐ |
| CDN integration for public assets | ☐ |
| Local storage fallback (dev) | ☐ |
| SAS token configuration | ☐ |
| Integration tests (mock Azure) | ☐ |

**Definition of Done:** Images stored in Azure Blob, retrievable via signed URLs, with local fallback for development.

---

### Sprint 5 – AI Prediction Pipeline

**Goal:** Core skin disease classification

| Task | Status |
|------|--------|
| AI provider abstraction | ☐ |
| Azure Custom Vision integration | ☐ |
| Local ONNX Runtime fallback | ☐ |
| Image preprocessing (resize, normalize) | ☐ |
| Prediction service & endpoint | ☐ |
| Confidence scoring & thresholds | ☐ |
| Top-K predictions response | ☐ |
| Model versioning | ☐ |
| Load testing | ☐ |

**Definition of Done:** `/api/v1/analyses` returns structured predictions with confidence scores.

---

### Sprint 6 – Explainability

**Goal:** Visual explanations for predictions

| Task | Status |
|------|--------|
| Grad-CAM implementation | ☐ |
| Heatmap overlay generation | ☐ |
| Explanation storage (heatmap images) | ☐ |
| Explainability endpoint | ☐ |
| LIME integration (optional) | ☐ |
| Frontend heatmap display spec | ☐ |
| Performance optimization | ☐ |
| Tests for explanation quality | ☐ |

**Definition of Done:** Each prediction includes Grad-CAM heatmap accessible via API.

---

### Sprint 7 – Chatbot

**Goal:** AI-powered patient guidance

| Task | Status |
|------|--------|
| Azure OpenAI / Llama integration | ☐ |
| Chat session management | ☐ |
| Context-aware prompts (analysis + history) | ☐ |
| Safety guardrails (medical disclaimer) | ☐ |
| Streaming responses | ☐ |
| Chat history persistence | ☐ |
| Rate limiting | ☐ |
| Feedback collection | ☐ |

**Definition of Done:** Users can chat about their analysis results with context-aware responses.

---

### Sprint 8 – Frontend

**Goal:** Functional web interface

| Task | Status |
|------|--------|
| React + TypeScript + Vite setup | ☐ |
| Tailwind + shadcn/ui components | ☐ |
| Authentication pages | ☐ |
| Image upload with drag-drop | ☐ |
| Analysis results display | ☐ |
| Heatmap visualization | ☐ |
| Chat interface | ☐ |
| Responsive design | ☐ |
| E2E tests (Playwright) | ☐ |

**Definition of Done:** End-to-end user flow works in browser.

---

### Sprint 9 – Deployment

**Goal:** Production deployment pipeline

| Task | Status |
|------|--------|
| Dockerfile (multi-stage) | ☐ |
| Docker Compose (local dev) | ☐ |
| GitHub Actions CI pipeline | ☐ |
| Azure Container Apps deployment | ☐ |
| Infrastructure as Code (Bicep/Terraform) | ☐ |
| Environment promotion (dev/staging/prod) | ☐ |
| Secrets management (Key Vault) | ☐ |
| Database migration automation | ☐ |
| Rollback strategy | ☐ |
| Load testing in staging | ☐ |

**Definition of Done:** Automated deploy to Azure, zero-downtime releases.

---

### Sprint 10 – Monitoring

**Goal:** Production observability

| Task | Status |
|------|--------|
| OpenTelemetry instrumentation | ☐ |
| Distributed tracing | ☐ |
| Prometheus metrics endpoint | ☐ |
| Grafana dashboards | ☐ |
| Structured log aggregation | ☐ |
| Alert rules (latency, errors, saturation) | ☐ |
| SLO definitions | ☐ |
| Synthetic monitoring | ☐ |
| Incident runbooks | ☐ |
| Cost monitoring | ☐ |

**Definition of Done:** Full observability stack operational, alerts tuned.

---

## Milestone Timeline

```
Q3 2026          Q4 2026          Q1 2027
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ S1 Foundation    │ S2 Auth         │ S3 Upload       │
├─────────────┤  ├─────────────┤  ├─────────────┤
│ S4 Azure Storage │ S5 AI Pipeline  │ S6 Explainability│
├─────────────┤  ├─────────────┤  ├─────────────┤
│ S7 Chatbot       │ S8 Frontend     │ S9 Deploy       │
├─────────────┤  ├─────────────┤  ├─────────────┤
│ S10 Monitoring   │                  │               │
└─────────────┘  └─────────────┘  └─────────────┘
```

---

## Risk Register

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Model accuracy below clinical threshold | High | Medium | Dedicated AI evaluation phase; fallback to specialist referral |
| Azure costs exceed budget | Medium | Medium | Cost monitoring from Sprint 1; spending alerts |
| Regulatory compliance (HIPAA/GDPR) | High | Low | Design for compliance from Sprint 2; legal review before AI rollout |
| Frontend/backend integration delays | Medium | Medium | Contract-first API; mock server for frontend dev |
| AI inference latency | High | Low | Local ONNX fallback; model optimization before production |

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| API p99 latency | < 500ms | Prometheus histogram |
| Model F1 score (weighted) | > 0.85 | Evaluation dataset |
| Upload success rate | > 99.5% | Application logs |
| Chatbot helpfulness rating | > 4.0/5 | User feedback |
| Deployment frequency | Weekly | GitHub Actions |
| Mean time to recovery | < 30min | Incident tracking |