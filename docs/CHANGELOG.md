# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-08-07

### Added

- Initial FastAPI application structure (`backend/app/main.py`)
- Health check endpoint (`GET /health`)
- Root endpoint (`GET /`) with service metadata
- Project folder architecture:
  - `api/` (v1 routers, dependencies)
  - `core/` (application core)
  - `config/` (configuration management)
  - `database/` (database layer)
  - `models/` (domain models)
  - `schemas/` (Pydantic schemas)
  - `repositories/` (data access layer)
  - `services/` (business logic)
  - `storage/` (file storage abstraction)
  - `ai/` (AI/ML integration: adapters, providers, prompts)
  - `exceptions/` (custom exceptions)
  - `logging/` (structured logging)
  - `middleware/` (HTTP middleware)
  - `utils/` (shared utilities)
  - `tests/` (test suite)
  - `constants/` (application constants)
- Documentation files:
  - `docs/ARCHITECTURE.md` - Comprehensive engineering handbook
  - `docs/ROADMAP.md` - Project roadmap with 10 sprints
  - `docs/SPRINTS.md` - Sprint tracker with detailed tasks
  - `docs/CHANGELOG.md` - This file
- `app/README.md` - Folder structure reference

### Changed

- None

### Deprecated

- None

### Removed

- None

### Fixed

- None

### Security

- None

---

## [Unreleased]

### Added

- None

### Changed

- None

### Deprecated

- None

### Removed

- None

### Fixed

- None

### Security

- None