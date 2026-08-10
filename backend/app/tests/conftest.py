"""Shared test configuration.

Settings now require a JWT signing secret from the environment, so provide a
test-only value before any module imports `get_settings()`.
"""

import os

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-that-is-long-enough-32")
