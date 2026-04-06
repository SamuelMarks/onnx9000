"""Provide an OpenAI-compatible API server."""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class OpenAIServer:
    """OpenAI API compatible server."""

    def __init__(self, port: int = 8000) -> None:
        """Initialize the instance."""
        self.port = port
        self.routes: Dict[str, Any] = {}
        self.is_running = False

    def add_route(self, path: str, handler: Any) -> None:
        """Add an API route."""
        self.routes[path] = handler

    def serve(self, cors: bool = True) -> bool:
        """Start the server."""
        self.is_running = True
        logger.info(f"Serving OpenAI API on port {self.port} (CORS: {cors})")
        return True
