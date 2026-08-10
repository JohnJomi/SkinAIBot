# SkinAIBot

A medical AI platform for skin disease diagnosis and analysis. 
Built with a separated Two-Team architecture: Application (Full Stack) and AI/ML Platform.

## Architecture

See `docs/ARCHITECTURE.md` and `docs/ROADMAP.md` for project structure and team responsibilities.

## Local Development

### Required configuration

The backend refuses to start without a JWT signing secret, and Compose fails
fast if it is not present in your environment. Generate one per environment and
never commit it:

```bash
export JWT_SECRET_KEY=$(openssl rand -hex 32)
```

See `backend/.env.example` for the full list of supported settings.

### Start the stack

```bash
docker compose up --build
```

Database migrations run automatically: a one-shot `migrate` service applies
`alembic upgrade head` once Postgres is healthy, and the backend only starts
after it completes. A fresh volume is therefore migrated to head on first boot.

- **Frontend**: http://localhost:3000
- **Application Backend**: http://localhost:8000
- **AI Service (Mock)**: http://localhost:8001
- **PostgreSQL**: postgres:5432 (inside the Docker network; not published to the host)

### Migrations

Migrations are applied automatically on startup (see above). To create a new one
after changing a model, mount the versions directory for that one command so the
generated file lands on the host instead of inside the throwaway container:

```bash
docker compose run --rm -v "$PWD/backend/alembic/versions:/app/backend/alembic/versions" migrate alembic revision --autogenerate -m "describe change"
```

The new file appears in `backend/alembic/versions/`. Review it before committing:
autogenerate does not detect every change, and on Linux the file is created as
root, so you may need `sudo chown "$USER"` on it.

Apply pending migrations without restarting the stack:

```bash
docker compose run --rm migrate alembic upgrade head
```
