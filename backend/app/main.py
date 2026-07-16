from fastapi import FastAPI

app = FastAPI(
    title="Skin Disease Diagnosis API",
    description="Backend API for the Skin Disease Diagnosis AI Platform.",
    version="0.1.0",
)


@app.get("/")
def read_root() -> dict[str, str]:
    return {
        "service": "Skin Disease Diagnosis API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy", "version": "0.1.0"}
