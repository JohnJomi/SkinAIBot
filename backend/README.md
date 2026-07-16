# Project Overview

This repository contains the first backend scaffold for the Skin Disease Diagnosis API.

# Project Structure

- `app/main.py`: FastAPI application entrypoint.
- `app/api/v1/routers/`: API router package reserved for future route modules.
- `app/core/`: Core application package reserved for shared core utilities.
- `app/middleware/`: Reserved for future middleware.
- `app/services/`: Reserved for service-layer code.
- `app/repositories/`: Reserved for repository-layer code.
- `app/models/`: Reserved for future data models.
- `app/schemas/`: Reserved for future request and response schemas.
- `app/dependencies/`: Reserved for dependency injection helpers.
- `app/utils/`: Reserved for shared utility helpers.
- `app/constants/`: Reserved for application constants.
- `app/tests/`: Reserved for tests.
- `docs/`: Documentation files, including the implementation report.

# Requirements

Python 3.12+

# Installation

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

# Running

```bash
uvicorn app.main:app --reload
```

# Available Endpoints

- `GET /`
- `GET /health`
- `/docs`

# Development Roadmap

This is only Phase 1.1 of the backend.
