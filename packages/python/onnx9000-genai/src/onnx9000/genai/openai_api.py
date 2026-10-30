"""Provide an OpenAI-compatible API server."""

import logging

logger = logging.getLogger(__name__)


class OpenAIServer:
    """OpenAI API compatible server."""

    def __init__(self, port: int = 8000) -> None:
        """Initialize OpenAIServer with configured parameters and state.

        Args:
            port: Parameter specifying port (int).
        """
        self.port = port
        self.routes: dict[str, int | float | str | bool | bytes | list | dict | tuple | None] = {}
        self.is_running = False

    def add_route(
        self, path: str, handler: int | float | str | bool | bytes | list | dict | tuple | None
    ) -> None:
        """Add an API route."""
        self.routes[path] = handler

    def serve(self, cors: bool = True) -> bool:
        """Start the server."""
        self.is_running = True
        logger.info(f"Serving OpenAI API on port {self.port} (CORS: {cors})")
        return True
