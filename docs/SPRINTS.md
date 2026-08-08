# Sprint Tracker

> Sprint numbering matches `docs/ROADMAP.md`. Status legend: ☐ Pending · 🟢 Ready to Start · 🔄 In Progress · ✅ Complete · 🔴 Blocked.

---

## Sprint 1 – Backend Foundation

### Objective
Establish production-ready backend infrastructure: configuration, database, logging, exceptions, and testing foundation.

### Scope
- Pydantic Settings configuration management
- Async SQLAlchemy database layer with Alembic migrations
- Structured JSON logging with correlation IDs
- Exception hierarchy and global handlers
- Request ID middleware
- Pre-commit hooks (ruff, mypy)
- Pytest infrastructure with fixtures

### Tasks

| ID | Task | Assignee | Status | Notes |
|----|------|----------|--------|-------|
| 1.1 | Create `config/settings.py` with BaseSettings | | ☐ Pending | Env-based config, validation |
| 1.2 | Implement `database/engine.py` (async engine) | | ☐ Pending | Pool config, echo flag |
| 1.3 | Create `database/session.py` (get_session dependency) | | ☐ Pending | AsyncSession, transaction mgmt |
| 1.4 | Add `database/base.py` (DeclarativeBase) | | ☐ Pending | Timestamp mixin |
| 1.5 | Configure Alembic (env.py, script.py.mako) | | ☐ Pending | Auto-generate migrations |
| 1.6 | Build exception hierarchy in `exceptions/` | | ☐ Pending | AppException + subclasses |
| 1.7 | Add global exception handlers in `exceptions/handlers.py` | | ☐ Pending | JSON error responses |
| 1.8 | Implement structured logging in `logging/` | | ☐ Pending | JSON formatter, contextvars |
| 1.9 | Create request ID middleware in `middleware/` | | ☐ Pending | UUID per request, log context |
| 1.10 | Add health check dependencies (DB, storage) | | ☐ Pending | `/health` detailed endpoint |
| 1.11 | Configure pyproject.toml (ruff, mypy, pytest) | | ☐ Pending | Line length 100, strict mode |
| 1.12 | Add pre-commit hooks | | ☐ Pending | ruff, mypy, check-yaml |
| 1.13 | Create pytest fixtures (session, client) | | ☐ Pending | `tests/conftest.py` |
| 1.14 | Write smoke tests for health/config | | ☐ Pending | `tests/test_health.py` |

### Completed

- [ ] Project folder structure created
- [ ] ARCHITECTURE.md, ROADMAP.md, SPRINTS.md, CHANGELOG.md created

### Pending

- [ ] All tasks in table above

### Blockers

- None currently

### Technical Debt

- None yet (greenfield)

### Review Notes

- Review config validation strategy
- Confirm async session lifecycle
- Verify logging output format matches observability requirements

### Next Sprint

Sprint 2 – Authentication

---

## Sprint 2 – Authentication

### Objective
Implement secure authentication with JWT access tokens, refresh token rotation, and role-based access control.

### Scope
- User model, schemas, repository
- Password hashing (argon2/bcrypt)
- JWT tokens (RS256) with JWKS endpoint
- Refresh token rotation with Redis blacklist
- Auth endpoints: register, login, logout, refresh, password reset
- Dependencies: `get_current_user`, `require_admin`
- Integration tests

### Tasks

| ID | Task | Assignee | Status | Notes |
|----|------|----------|--------|-------|
| 2.1 | Create User SQLAlchemy model | | ☐ Pending | email, hashed_pw, is_active, roles |
| 2.2 | Create User Pydantic schemas | | ☐ Pending | Create, Update, Response, Public |
| 2.3 | Implement UserRepository | | ☐ Pending | CRUD + get_by_email |
| 2.4 | Add password hashing utility | | ☐ Pending | argon2-cffi / bcrypt |
| 2.5 | Implement JWT service (encode/decode) | | ☐ Pending | RS256, private/public keys |
| 2.6 | Create token schemas (access, refresh) | | ☐ Pending | TypedToken, payload models |
| 2.7 | Build Redis token blacklist | | ☐ Pending | TTL = access token expiry |
| 2.8 | Create auth endpoints (router) | | ☐ Pending | /auth/register, /login, /refresh, /logout |
| 2.9 | Add password reset flow (email token) | | ☐ Pending | Mock email service for now |
| 2.10 | Implement `get_current_user` dependency | | ☐ Pending | Validate access token, fetch user |
| 2.11 | Implement `require_admin` dependency | | ☐ Pending | Role check |
| 2.12 | Add JWKS endpoint for public keys | | ☐ Pending | /.well-known/jwks.json |
| 2.13 | Write integration tests for auth flow | | ☐ Pending | Register → login → access protected |

### Completed

- [ ] None

### Pending

- [ ] All tasks in table above

### Blockers

- [ ] Redis instance required (local/docker)
- [ ] RSA key generation for JWT signing

### Technical Debt

- [ ] Email service abstraction needed for password reset

### Review Notes

- Security review of token handling
- Verify refresh token rotation prevents replay
- Check rate limiting on auth endpoints

### Next Sprint

Sprint 3 – Image Upload

---

## Sprint 3 – Image Upload

### Objective
Enable secure, validated image uploads that create analysis records for downstream AI processing.

### Scope
- Upload endpoint (multipart)
- Presigned URL + direct upload flows
- File validation: MIME, size, magic bytes
- Upload status tracking
- Image preprocessing pipeline
- Analysis record creation

### Tasks

| ID | Task | Assignee | Status | Notes |
|----|------|----------|--------|-------|
| 3.1 | Create image schemas (UploadRequest, Response) | | ☐ Pending | Pydantic |
| 3.2 | Add file validation utilities | | ☐ Pending | python-magic, size limits (10MB) |
| 3.3 | Build upload router/endpoints | | ☐ Pending | POST /uploads/direct, POST /uploads/presign |
| 3.4 | Add image preprocessing service | | ☐ Pending | PIL: resize 224x224, normalize |
| 3.5 | Implement upload status tracking | | ☐ Pending | Analysis record creation |
| 3.6 | Add MIME/extension allowlist | | ☐ Pending | jpg, jpeg, png, webp |
| 3.7 | Handle duplicate/oversize uploads | | ☐ Pending | 413 / 409 responses |
| 3.8 | Write integration tests | | ☐ Pending | Validation, both upload flows |

### Completed

- [ ] None

### Pending

- [ ] All tasks in table above

### Blockers

- [ ] Storage backend availability (Sprint 4 provides it)

### Technical Debt

- [ ] Virus scanning integration (future)
- [ ] Upload resumability for large files (future)

### Review Notes

- Verify validation rejects malformed files
- Test both presigned and direct flows
- Confirm error responses match API contract

### Next Sprint

Sprint 4 – Azure Storage

---

## Sprint 4 – Azure Storage

### Objective
Integrate Azure Blob Storage with a clean storage abstraction and presigned URL generation, plus local fallback for development.

### Scope
- Storage abstraction interface
- Azure Blob Storage implementation
- Presigned URL generation (PUT/GET)
- Container & lifecycle policies
- CDN integration for public assets
- Local storage fallback for development

### Tasks

| ID | Task | Assignee | Status | Notes |
|----|------|----------|--------|-------|
| 4.1 | Define `StorageBackend` abstract base class | | ☐ Pending | upload, download, delete, presign |
| 4.2 | Implement `AzureBlobStorage` backend | | ☐ Pending | DefaultAzureCredential |
| 4.3 | Implement `LocalStorage` backend (dev) | | ☐ Pending | Filesystem-based |
| 4.4 | Create storage factory/dependency | | ☐ Pending | Config-driven selection |
| 4.5 | Add presigned URL generation | | ☐ Pending | PUT (upload) + GET (download) |
| 4.6 | Configure container & lifecycle policies | | ☐ Pending | Tiering, retention |
| 4.7 | Add CDN integration for public heatmaps | | ☐ Pending | Azure CDN / Front Door |
| 4.8 | Write integration tests (mock Azure) | | ☐ Pending | Presign flow, backend swap |

### Completed

- [ ] None

### Pending

- [ ] All tasks in table above

### Blockers

- [ ] Azure Storage account credentials
- [ ] Container SAS policy configuration

### Technical Debt

- [ ] Geo-redundancy configuration (future)
- [ ] Storage cost optimization per tier (future)

### Review Notes

- Verify presigned URL expiry bounds
- Confirm CORS configuration on storage account
- Test local fallback parity with Azure

### Next Sprint

Sprint 5 – AI Prediction Pipeline

---

## Sprint 5 – AI Prediction Pipeline

### Objective
Build the core skin disease classification pipeline with provider abstraction for Azure Custom Vision and local ONNX fallback.

### Scope
- AI provider abstraction interface
- Azure Custom Vision provider
- Local ONNX Runtime provider
- Image preprocessing service
- Prediction service orchestration
- `/api/v1/analyses` endpoint
- Model versioning support
- Confidence thresholds

### Tasks

| ID | Task | Assignee | Status | Notes |
|----|------|----------|--------|-------|
| 5.1 | Define `AIProvider` abstract base class | | ☐ Pending | predict(), explain() |
| 5.2 | Implement `AzureCustomVisionProvider` | | ☐ Pending | REST API client |
| 5.3 | Implement `ONNXRuntimeProvider` | | ☐ Pending | Local .onnx model |
| 5.4 | Create provider factory/dependency | | ☐ Pending | Config-driven |
| 5.5 | Build preprocessing pipeline | | ☐ Pending | Resize, normalize, tensor conversion |
| 5.6 | Create prediction schemas | | ☐ Pending | PredictionResult, AnalysisResponse |
| 5.7 | Implement AnalysisService | | ☐ Pending | Orchestrate upload → preprocess → predict |
| 5.8 | Add confidence threshold config | | ☐ Pending | Per-class thresholds |
| 5.9 | Create analyses router | | ☐ Pending | POST /analyses, GET /analyses/{id} |
| 5.10 | Implement model version tracking | | ☐ Pending | Response includes model_version |
| 5.11 | Write unit tests for providers | | ☐ Pending | Mock HTTP, test preprocessing |

### Completed

- [ ] None

### Pending

- [ ] All tasks in table above

### Blockers

- [ ] Azure Custom Vision project + prediction key
- [ ] Trained ONNX model file
- [ ] Class labels mapping

### Technical Debt

- [ ] Batch prediction endpoint (future)
- [ ] A/B testing framework (future)

### Review Notes

- Validate preprocessing matches training
- Confirm provider failover behavior
- Benchmark latency (target < 500ms)

### Next Sprint

Sprint 6 – Explainability

---

## Sprint 6 – Explainability

### Objective
Generate Grad-CAM heatmaps for model predictions to provide visual explanations.

### Scope
- Grad-CAM implementation (Captum or custom)
- Heatmap overlay generation
- Heatmap storage and retrieval
- Explainability endpoint
- Frontend integration spec

### Tasks

| ID | Task | Assignee | Status | Notes |
|----|------|----------|--------|-------|
| 6.1 | Implement Grad-CAM for ONNX models | | ☐ Pending | Captum or manual gradients |
| 6.2 | Add Azure Custom Vision explanation support | | ☐ Pending | If API supports it |
| 6.3 | Create heatmap overlay utility | | ☐ Pending | PIL: colormap + alpha blend |
| 6.4 | Store heatmaps in blob storage | | ☐ Pending | processed/{user_id}/{id}_heatmap.jpg |
| 6.5 | Add explanation to AnalysisResponse | | ☐ Pending | heatmap_url field |
| 6.6 | Create explainability endpoint | | ☐ Pending | GET /analyses/{id}/explanation |
| 6.7 | Optimize heatmap generation latency | | ☐ Pending | Target < 200ms additional |
| 6.8 | Write tests for heatmap quality | | ☐ Pending | Shape checks, visual regression |

### Completed

- [ ] None

### Pending

- [ ] All tasks in table above

### Blockers

- [ ] Access to model internals (gradients) for ONNX
- [ ] Azure Custom Vision explainability API availability

### Technical Debt

- [ ] LIME integration as alternative (future)

### Review Notes

- Verify heatmap aligns with clinical regions
- Test with various image sizes
- Confirm storage costs acceptable

### Next Sprint

Sprint 7 – Chatbot

---

## Sprint 7 – Chatbot

### Objective
Implement context-aware chatbot for patient guidance on analysis results.

### Scope
- LLM provider integration (Azure OpenAI)
- Chat session/message models
- Context injection (analysis + history)
- Safety guardrails (medical disclaimer)
- Streaming responses
- Feedback collection

### Tasks

| ID | Task | Assignee | Status | Notes |
|----|------|----------|--------|-------|
| 7.1 | Create ChatSession, ChatMessage models | | ☐ Pending | SQLAlchemy |
| 7.2 | Add chat schemas (Create, Response) | | ☐ Pending | Pydantic |
| 7.3 | Implement LLM provider abstraction | | ☐ Pending | chat(), stream() |
| 7.4 | Build Azure OpenAI provider | | ☐ Pending | GPT-4o / GPT-4o-mini |
| 7.5 | Design system prompt template | | ☐ Pending | Medical disclaimer, context rules |
| 7.6 | Implement context assembly service | | ☐ Pending | Analysis + history → messages |
| 7.7 | Create chat router endpoints | | ☐ Pending | Sessions, messages, streaming |
| 7.8 | Add rate limiting per user | | ☐ Pending | Redis sliding window |
| 7.9 | Implement feedback endpoint | | ☐ Pending | Thumbs up/down per message |
| 7.10 | Write integration tests | | ☐ Pending | Context injection, streaming |

### Completed

- [ ] None

### Pending

- [ ] All tasks in table above

### Blockers

- [ ] Azure OpenAI deployment + quota
- [ ] Prompt engineering iteration time

### Technical Debt

- [ ] Local LLM fallback (Ollama) for cost control
- [ ] Conversation summarization for long histories

### Review Notes

- Safety review of prompt templates
- Verify no PHI in logs
- Test edge cases (empty analysis, errors)

### Next Sprint

Sprint 8 – Frontend

---

## Sprint 8 – Frontend

### Objective
Deliver functional React frontend for the complete user journey.

### Scope
- React + TypeScript + Vite setup
- Authentication flows
- Image upload with preview
- Analysis results with heatmap
- Chat interface
- Responsive design

### Tasks

| ID | Task | Assignee | Status | Notes |
|----|------|----------|--------|-------|
| 8.1 | Initialize React + TS + Vite project | | ☐ Pending | ESLint, Prettier, Vitest |
| 8.2 | Configure Tailwind + shadcn/ui | | ☐ Pending | Component library |
| 8.3 | Generate API client from OpenAPI | | ☐ Pending | orval / openapi-fetch |
| 8.4 | Build auth pages (login, register) | | ☐ Pending | React Router, form validation |
| 8.5 | Create upload component (drag-drop) | | ☐ Pending | Presigned URL flow |
| 8.6 | Build analysis results view | | ☐ Pending | Predictions table, confidence bars |
| 8.7 | Implement heatmap overlay display | | ☐ Pending | Image + canvas overlay |
| 8.8 | Build chat interface | | ☐ Pending | Streaming messages, markdown |
| 8.9 | Add responsive layout + navigation | | ☐ Pending | Mobile-first |
| 8.10 | Write E2E tests (Playwright) | | ☐ Pending | Critical user flows |
| 8.11 | Configure CI for frontend | | ☐ Pending | Build, lint, test |

### Completed

- [ ] None

### Pending

- [ ] All tasks in table above

### Blockers

- [ ] Backend API stability (Sprints 2-7 complete)
- [ ] Design system decisions

### Technical Debt

- [ ] Storybook for component documentation
- [ ] Visual regression testing

### Review Notes

- Accessibility audit (WCAG AA)
- Cross-browser testing
- Performance budget (< 3s TTI)

### Next Sprint

Sprint 9 – Deployment

---

## Sprint 9 – Deployment

### Objective
Automate deployment to Azure with zero-downtime releases.

### Scope
- Multi-stage Dockerfile
- GitHub Actions CI/CD
- Azure Container Apps deployment
- Infrastructure as Code (Bicep)
- Environment promotion
- Secrets management

### Tasks

| ID | Task | Assignee | Status | Notes |
|----|------|----------|--------|-------|
| 9.1 | Create multi-stage Dockerfile | | ☐ Pending | Builder → runtime, non-root |
| 9.2 | Add docker-compose.yml (local) | | ☐ Pending | API, DB, Redis, Azurite |
| 9.3 | Build GitHub Actions CI workflow | | ☐ Pending | Lint, typecheck, test, build |
| 9.4 | Create CD workflow (staging) | | ☐ Pending | Deploy on main branch |
| 9.5 | Create CD workflow (production) | | ☐ Pending | Manual approval, tag trigger |
| 9.6 | Write Bicep modules (ACA, DB, Storage) | | ☐ Pending | Parameterized environments |
| 9.7 | Configure Azure Key Vault integration | | ☐ Pending | Secrets → container env |
| 9.8 | Set up managed identities | | ☐ Pending | ACA → Azure resources |
| 9.9 | Implement DB migration in startup | | ☐ Pending | Alembic upgrade head |
| 9.10 | Configure custom domain + TLS | | ☐ Pending | Azure Front Door / App Gateway |
| 9.11 | Load test staging environment | | ☐ Pending | k6 or Locust |

### Completed

- [ ] None

### Pending

- [ ] All tasks in table above

### Blockers

- [ ] Azure subscription + permissions
- [ ] Container registry setup

### Technical Debt

- [ ] Blue-green deployment (future)
- [ ] Database backup/restore automation

### Review Notes

- Security scan container images
- Verify secret rotation works
- Test rollback procedure

### Next Sprint

Sprint 10 – Monitoring

---

## Sprint 10 – Monitoring

### Objective
Establish comprehensive observability for production operations.

### Scope
- OpenTelemetry instrumentation
- Distributed tracing
- Prometheus metrics + Grafana
- Structured log aggregation
- Alerting rules
- SLO definitions
- Runbooks

### Tasks

| ID | Task | Assignee | Status | Notes |
|----|------|----------|--------|-------|
| 10.1 | Add OpenTelemetry SDK + auto-instrument | | ☐ Pending | FastAPI, SQLAlchemy, Redis, HTTPX |
| 10.2 | Configure OTLP exporter (Azure Monitor) | | ☐ Pending | Traces, metrics, logs |
| 10.3 | Define custom metrics (business + technical) | | ☐ Pending | Request latency, error rate, predictions |
| 10.4 | Build Grafana dashboards | | ☐ Pending | RED metrics, business KPIs |
| 10.5 | Set up log aggregation (Azure Log Analytics) | | ☐ Pending | Structured JSON parsing |
| 10.6 | Create alert rules | | ☐ Pending | Latency > 1s, error rate > 1%, saturation |
| 10.7 | Define SLOs/SLIs | | ☐ Pending | Availability 99.9%, latency p99 < 500ms |
| 10.8 | Configure synthetic monitoring | | ☐ Pending | Health endpoint, critical flows |
| 10.9 | Write incident runbooks | | ☐ Pending | Common scenarios + escalation |
| 10.10 | Enable cost monitoring alerts | | ☐ Pending | Daily spend > threshold |
| 10.11 | Conduct game day exercise | | ☐ Pending | Simulate outage, test runbooks |

### Completed

- [ ] None

### Pending

- [ ] All tasks in table above

### Blockers

- [ ] Azure Monitor workspace
- [ ] Grafana Cloud / self-hosted

### Technical Debt

- [ ] Continuous profiling (py-spy)
- [ ] User session replay (future)

### Review Notes

- Alert fatigue prevention (tune thresholds)
- Dashboard usability review
- On-call rotation readiness

### Next Steps

- Post-launch iteration
- Feature backlog prioritization
- Technical debt sprint

---

## Sprint Template (for future sprints)

```markdown
## Sprint N – [Name]

### Objective
[One sentence goal]

### Scope
- [Bullet points of what's included]

### Tasks

| ID | Task | Assignee | Status | Notes |
|----|------|----------|--------|-------|
| N.1 | [Task description] | | ☐ Pending | [Details] |

### Completed
- [ ] Items done

### Pending
- [ ] Items remaining

### Blockers
- [ ] External dependencies

### Technical Debt
- [ ] Known shortcuts

### Review Notes
- [ ] Items for sprint review

### Next Sprint
Sprint N+1 – [Name]
```