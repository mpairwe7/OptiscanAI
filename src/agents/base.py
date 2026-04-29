"""Base agent class with lifecycle management, tool execution, and event integration."""
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine

from src.agents.event_bus import Event, EventBus, EventType, event_bus

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from a tool invocation."""
    tool: str
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    elapsed_ms: float = 0.0


@dataclass
class AgentState:
    """Observable state of an agent."""
    name: str
    status: str = "idle"  # idle | running | error | stopped
    last_action: str = ""
    last_action_at: str = ""
    actions_taken: int = 0
    errors: int = 0
    started_at: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "last_action": self.last_action,
            "last_action_at": self.last_action_at,
            "actions_taken": self.actions_taken,
            "errors": self.errors,
            "started_at": self.started_at,
        }


# Tool type: async function returning ToolResult
Tool = Callable[..., Coroutine[Any, Any, ToolResult]]


class BaseAgent(ABC):
    """Base class for RetinalAI autonomous agents.

    Each agent:
    - Has a name and description
    - Registers tools (functions it can call)
    - Subscribes to events on the bus
    - Runs an optional periodic loop
    - Tracks its own state for observability
    """

    def __init__(self, name: str, bus: EventBus | None = None):
        self.name = name
        self.bus = bus or event_bus
        self.state = AgentState(name=name)
        self._tools: dict[str, Tool] = {}
        self._task: asyncio.Task | None = None

    # ── Tool registration ──

    def register_tool(self, name: str, fn: Tool):
        """Register an async tool function."""
        self._tools[name] = fn

    async def use_tool(self, name: str, **kwargs) -> ToolResult:
        """Invoke a registered tool by name."""
        tool_fn = self._tools.get(name)
        if not tool_fn:
            return ToolResult(tool=name, success=False, error=f"Unknown tool: {name}")

        import time
        t0 = time.perf_counter()
        try:
            result = await tool_fn(**kwargs)
            result.elapsed_ms = (time.perf_counter() - t0) * 1000
            self.state.actions_taken += 1
            self.state.last_action = name
            self.state.last_action_at = datetime.now(timezone.utc).isoformat()
            return result
        except Exception as e:
            self.state.errors += 1
            logger.error(f"Agent {self.name} tool {name} failed: {e}", exc_info=True)
            return ToolResult(
                tool=name,
                success=False,
                error=str(e),
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

    # ── Event integration ──

    def subscribe(self, event_type: EventType, handler):
        """Subscribe to an event type."""
        self.bus.subscribe(event_type, handler)

    async def emit(self, event_type: EventType, data: dict | None = None):
        """Emit an event from this agent."""
        await self.bus.emit(Event(
            type=event_type,
            source=self.name,
            data=data or {},
        ))

    # ── Lifecycle ──

    async def start(self):
        """Start the agent. Calls setup() then begins the run loop if defined."""
        self.state.status = "running"
        self.state.started_at = datetime.now(timezone.utc).isoformat()
        logger.info(f"Agent {self.name} starting")

        await self.setup()
        await self.emit(EventType.AGENT_STARTED, {"agent": self.name})

        interval = self.loop_interval_seconds()
        if interval and interval > 0:
            self._task = asyncio.create_task(self._run_loop(interval))

    async def stop(self):
        """Stop the agent gracefully."""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.state.status = "stopped"
        await self.emit(EventType.AGENT_STOPPED, {"agent": self.name})
        logger.info(f"Agent {self.name} stopped")

    async def _run_loop(self, interval: float):
        """Periodic execution loop."""
        while self.state.status == "running":
            try:
                await self.tick()
            except Exception as e:
                self.state.errors += 1
                self.state.status = "error"
                logger.error(f"Agent {self.name} tick failed: {e}", exc_info=True)
                await self.emit(EventType.AGENT_ERROR, {
                    "agent": self.name,
                    "error": str(e),
                })
                self.state.status = "running"

            await asyncio.sleep(interval)

    # ── Abstract methods for subclasses ──

    @abstractmethod
    async def setup(self):
        """Initialize agent: register tools, subscribe to events."""
        ...

    async def tick(self):
        """Called periodically if loop_interval_seconds() > 0. Override in subclasses."""
        pass

    def loop_interval_seconds(self) -> float | None:
        """Return interval for periodic tick, or None to disable loop."""
        return None

    @property
    def tools_available(self) -> list[str]:
        return list(self._tools.keys())
