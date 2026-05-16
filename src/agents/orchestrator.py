"""AgentOrchestrator: Manages the lifecycle of all RetinalAI agents.

Starts, stops, and monitors the three autonomous agents. Provides a unified
interface for the backend API to query agent state and trigger actions.
"""

import logging
from typing import Optional

from src.agents.base import BaseAgent
from src.agents.event_bus import EventBus, event_bus
from src.agents.governance_agent import GovernanceAgent
from src.agents.monitor_agent import MonitorAgent
from src.agents.screening_agent import ScreeningAgent

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Manages the full agent ensemble for the RetinalAI platform.

    Usage:
        orchestrator = AgentOrchestrator(model_service=model_service)
        await orchestrator.start()  # call in FastAPI lifespan
        ...
        await orchestrator.stop()   # call on shutdown
    """

    def __init__(
        self,
        model_service=None,
        review_gate=None,
        audit_trail=None,
        bus: Optional[EventBus] = None,
        monitor_interval: float = 60.0,
        governance_interval: float = 300.0,
    ):
        self.bus = bus or event_bus
        self._agents: dict[str, BaseAgent] = {}

        # Create agents
        if model_service:
            self._agents["screening"] = ScreeningAgent(
                model_service=model_service,
                review_gate=review_gate,
                bus=self.bus,
            )

        self._agents["monitor"] = MonitorAgent(
            tick_interval=monitor_interval,
            bus=self.bus,
        )

        self._agents["governance"] = GovernanceAgent(
            audit_trail=audit_trail,
            review_gate=review_gate,
            tick_interval=governance_interval,
            bus=self.bus,
        )

    async def start(self):
        """Start all agents."""
        logger.info(f"Starting {len(self._agents)} agents")
        for name, agent in self._agents.items():
            try:
                await agent.start()
                logger.info(f"Agent '{name}' started")
            except Exception as e:
                logger.error(f"Failed to start agent '{name}': {e}", exc_info=True)

    async def stop(self):
        """Stop all agents gracefully."""
        logger.info("Stopping all agents")
        for name, agent in self._agents.items():
            try:
                await agent.stop()
            except Exception as e:
                logger.error(f"Failed to stop agent '{name}': {e}")

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        return self._agents.get(name)

    @property
    def screening(self) -> Optional[ScreeningAgent]:
        agent = self._agents.get("screening")
        return agent if isinstance(agent, ScreeningAgent) else None

    @property
    def monitor(self) -> Optional[MonitorAgent]:
        agent = self._agents.get("monitor")
        return agent if isinstance(agent, MonitorAgent) else None

    @property
    def governance(self) -> Optional[GovernanceAgent]:
        agent = self._agents.get("governance")
        return agent if isinstance(agent, GovernanceAgent) else None

    def status(self) -> dict:
        """Get status of all agents for the dashboard."""
        agents = {}
        for name, agent in self._agents.items():
            agents[name] = {
                **agent.state.to_dict(),
                "tools": agent.tools_available,
            }
        return {
            "total_agents": len(self._agents),
            "agents": agents,
            "event_bus": self.bus.stats,
        }
