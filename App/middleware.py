from fastapi_sessions import SessionMiddleware, SessionConfig, Backends
from fastapi_sessions.backends.implementations import InMemoryBackend

SECRET_KEY = "your-secret-key"  # Should be kept secret
ALGORITHM = "HS256"

session_backend = InMemoryBackend()

app.add_middleware(
    SessionMiddleware,
    config=SessionConfig(
        secret_key=SECRET_KEY,
        session_backend=session_backend,
        session_cookie="session_id",
    ),
)
