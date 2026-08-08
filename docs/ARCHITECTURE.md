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
