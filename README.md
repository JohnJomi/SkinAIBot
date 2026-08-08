# SkinAIBot

A medical AI platform for skin disease diagnosis and analysis. 
Built with a separated Two-Team architecture: Application (Full Stack) and AI/ML Platform.

## Architecture

See `docs/ARCHITECTURE.md` and `docs/ROADMAP.md` for project structure and team responsibilities.

## Local Development

Start the development environment using Docker Compose:

```bash
docker compose up --build
```

- **Frontend**: http://localhost:3000
- **Application Backend**: http://localhost:8000
- **AI Service (Mock)**: http://localhost:8001
- **PostgreSQL**: localhost:5432
