"""Turn every API failure into one shape the UI and `shai` can show: {"detail": "<sentence>"}.

Without this, bad input comes back as a list of objects (the UI showed "[object Object]")
and an unexpected crash comes back as a plain-text 500 page with no hint of what failed.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

log = logging.getLogger(__name__)

INTERNAL_ERROR_MESSAGE = (
    "Something went wrong inside StronghouseAI ({kind}). "
    "The details are in the terminal where the app is running."
)


def describe_validation_errors(errors: list[dict]) -> str:
    """One readable sentence per invalid field, e.g. 'context: Input should be greater than 0'."""
    parts = []
    for error in errors:
        location = [str(part) for part in error.get("loc", ()) if part not in ("body", "query")]
        field = ".".join(location)
        message = str(error.get("msg", "is not valid"))
        parts.append(f"{field}: {message}" if field else message)
    return "; ".join(parts) or "The request was not valid."


async def _validation_error(_request: Request, error: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422, content={"detail": describe_validation_errors(error.errors())}
    )


async def _unexpected_error(request: Request, error: Exception) -> JSONResponse:
    # The traceback goes to the log only; the message to the user names the error type,
    # never its text, so paths or tokens inside an exception message are not echoed back.
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    kind = type(error).__name__
    return JSONResponse(
        status_code=500, content={"detail": INTERNAL_ERROR_MESSAGE.format(kind=kind)}
    )


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, _validation_error)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, _unexpected_error)
