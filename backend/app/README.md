# App Directory Structure

This document explains the purpose of each folder in the `app/` directory.

## Core Application

- **api/** - API layer containing versioned endpoints and dependencies
  - **v1/** - API version 1 routes and routers
  - **dependencies/** - FastAPI dependency injection providers
- **core/** - Core application logic, configuration, and base classes
- **config/** - Application configuration management (settings, environment variables)
- **database/** - Database connection, session management, and migrations
- **models/** - Database models (SQLAlchemy/Pydantic)
- **schemas/** - Pydantic schemas for request/response validation and serialization
- **repositories/** - Data access layer (repository pattern for database operations)
- **services/** - Business logic layer (service classes orchestrating repositories)
- **utils/** - Shared utility functions and helpers

## Infrastructure

- **middleware/** - Custom FastAPI middleware (logging, error handling, etc.)
- **exceptions/** - Custom exception classes and exception handlers
- **logging/** - Logging configuration and formatters
- **storage/** - File storage abstractions (S3, local, etc.)

## AI/ML

- **ai/** - AI/ML related components
  - **adapters/** - Adapters for different AI model interfaces
  - **providers/** - AI provider implementations (OpenAI, Anthropic, Bedrock, etc.)
  - **prompts/** - Prompt templates and prompt engineering utilities

## Testing

- **tests/** - Test suite (unit, integration, e2e)