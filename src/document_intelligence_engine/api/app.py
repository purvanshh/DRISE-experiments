"""Compatibility shim for the top-level API app."""

from api.main import app, create_app

__all__ = ["app", "create_app"]
