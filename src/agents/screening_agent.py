"""ScreeningAgent: LangGraph + Claude powered clinical screening orchestrator.

Wraps the LangGraph screening pipeline (graph.py) as an agent with:
- Event bus integration (reacts to SCAN_RECEIVED, emits SCAN_ANALYZED)
- Lifecycle management (start/stop)
- State tracking for the dashboard
- Fallback to deterministic mode when Claude API is unavailable

The actual workflow is in graph.py — this agent manages when and how it runs.
"""
import logging
from typing import Any

from PIL import Image

from src.agents.base import BaseAgent, ToolResult
from src.agents.event_bus import EventType

logger = logging.getLogger(__name__)


class ScreeningAgent(BaseAgent):
    """Autonomous screening agent backed by a LangGraph StateGraph.

    Tools:
        run_screening: Execute the full LangGraph pipeline on an image
        get_last_report: Retrieve the most recent screening report
    """

    def __init__(self, model_service, review_gate=None, **kwargs):
        super().__init__(name="screening_agent", **kwargs)
        self.model_service = model_service
        self.review_gate = review_gate
        self._last_report: dict | None = None

    async def setup(self):
        self.register_tool("run_screening", self._run_screening)
        self.register_tool("get_last_report", self._get_last_report)

        # React to scans arriving on the event bus
        self.subscribe(EventType.SCAN_RECEIVED, self._on_scan_received)

    async def _on_scan_received(self, event):
        """Auto-run screening when a scan arrives on the bus."""
        image = event.data.get("image")
        scan_id = event.data.get("scan_id", event.event_id)
        if image:
            await self.use_tool("run_screening", image=image, scan_id=scan_id)

    async def _run_screening(self, image: Image.Image, scan_id: str = "") -> ToolResult:
        """Execute the LangGraph screening pipeline."""
        from src.agents.graph import run_screening

        try:
            report = await run_screening(image, scan_id=scan_id)
            self._last_report = report

            return ToolResult(
                tool="run_screening",
                success=True,
                data=report,
            )
        except Exception as e:
            logger.error(f"LangGraph screening failed: {e}", exc_info=True)
            return ToolResult(
                tool="run_screening",
                success=False,
                error=str(e),
            )

    async def _get_last_report(self) -> ToolResult:
        """Return the most recent screening report."""
        if self._last_report:
            return ToolResult(tool="get_last_report", success=True, data=self._last_report)
        return ToolResult(tool="get_last_report", success=False, error="No screening has been run yet")
