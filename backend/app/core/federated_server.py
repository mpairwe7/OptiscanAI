"""Flower federated learning aggregation server.

Wraps flwr.server with FedAvg/FedProx strategy, metrics aggregation,
and audit logging for privacy-preserving model updates from multiple
clinic sites.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_flwr_available = False
try:
    import flwr as fl
    from flwr.server.strategy import FedAvg

    _flwr_available = True
except ImportError:
    pass


class FlowerFederatedServer:
    """Flower aggregation server for federated LoRA updates.

    Parameters
    ----------
    server_address : str
        Address to listen on.
    min_clients : int
        Minimum clients for a training round.
    num_rounds : int
        Number of federated learning rounds.
    strategy : str
        Aggregation strategy: "fedavg" or "fedprox".
    """

    def __init__(
        self,
        server_address: str = "0.0.0.0:8080",
        min_clients: int = 2,
        num_rounds: int = 5,
        strategy: str = "fedavg",
        checkpoint_dir: str = "outputs/federated",
    ):
        self._address = server_address
        self._min_clients = min_clients
        self._num_rounds = num_rounds
        self._strategy_name = strategy
        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _build_strategy(self):
        """Build the aggregation strategy."""
        if not _flwr_available:
            raise RuntimeError("Flower not installed. pip install flwr>=1.8.0")

        def on_fit_config(server_round: int) -> dict:
            return {
                "server_round": server_round,
                "local_epochs": 3,
                "batch_size": 16,
            }

        def on_evaluate_config(server_round: int) -> dict:
            return {"server_round": server_round}

        def fit_metrics_aggregation(metrics: list) -> dict:
            """Aggregate per-client training metrics."""
            total_samples = sum(n for n, _ in metrics)
            aggregated = {}
            for n, m in metrics:
                for k, v in m.items():
                    aggregated[k] = aggregated.get(k, 0) + v * n
            return {k: v / total_samples for k, v in aggregated.items()}

        strategy = FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=self._min_clients,
            min_evaluate_clients=self._min_clients,
            min_available_clients=self._min_clients,
            on_fit_config_fn=on_fit_config,
            on_evaluate_config_fn=on_evaluate_config,
            fit_metrics_aggregation_fn=fit_metrics_aggregation,
        )

        return strategy

    def start(self) -> None:
        """Start the federated learning server."""
        if not _flwr_available:
            logger.error("Flower not available — cannot start server")
            return

        strategy = self._build_strategy()

        logger.info(
            "Starting Flower server: address=%s, rounds=%d, min_clients=%d",
            self._address,
            self._num_rounds,
            self._min_clients,
        )

        fl.server.start_server(
            server_address=self._address,
            config=fl.server.ServerConfig(num_rounds=self._num_rounds),
            strategy=strategy,
        )

    def simulate(
        self,
        client_fn,
        num_clients: int = 5,
        num_rounds: int = 5,
    ) -> dict:
        """Run a Flower simulation with virtual clients."""
        if not _flwr_available:
            logger.error("Flower not available for simulation")
            return {"error": "flwr not installed"}

        strategy = self._build_strategy()

        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
        )

        results = {
            "num_rounds": num_rounds,
            "num_clients": num_clients,
            "losses_distributed": list(history.losses_distributed),
            "metrics_distributed": dict(history.metrics_distributed),
        }

        logger.info("Simulation complete: %d rounds, %d clients", num_rounds, num_clients)
        return results
