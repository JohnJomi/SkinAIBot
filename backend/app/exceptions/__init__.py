"""Application exception hierarchy and FastAPI exception handlers."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class AIUpstreamError(Exception):
    """Base error for failures communicating with the AI service."""

    status_code = 502
    code = "AI_UPSTREAM_ERROR"

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class AIServiceUnavailableError(AIUpstreamError):
    """The AI service is unreachable or timed out."""

    status_code = 503
    code = "AI_SERVICE_UNAVAILABLE"


class AIUpstreamResponseError(AIUpstreamError):
    """The AI service returned an unexpected HTTP error."""

    status_code = 502
    code = "AI_UPSTREAM_ERROR"


async def ai_upstream_error_handler(
    request: Request, exc: AIUpstreamError
) -> JSONResponse:
    """Serialize AI upstream errors into the documented structured error format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
            }
        },
    )


class AppError(Exception):
    """Base error for application-level failures with a documented status code."""

    status_code = 400
    code = "APP_ERROR"

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class EmailAlreadyRegisteredError(AppError):
    status_code = 409
    code = "EMAIL_ALREADY_REGISTERED"


class InvalidCredentialsError(AppError):
    status_code = 401
    code = "INVALID_CREDENTIALS"


class PasswordTooLongError(AppError):
    status_code = 422
    code = "PASSWORD_TOO_LONG"


class InvalidFileTypeError(AppError):
    status_code = 422
    code = "INVALID_FILE_TYPE"


class FileTooLargeError(AppError):
    status_code = 413
    code = "FILE_TOO_LARGE"


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Serialize application errors into the documented structured error format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
            }
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register application exception handlers on the FastAPI app."""
    app.add_exception_handler(AIUpstreamError, ai_upstream_error_handler)
    app.add_exception_handler(AppError, app_error_handler)
