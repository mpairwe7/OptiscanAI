"""Federated learning client skeleton for cross-site retinal model training.

Phase 4 future-proofing module.  Provides abstract and concrete client
implementations for two federated frameworks:

* **Flower** (``FlowerRetinalClient``) -- fully functional client that
  wraps a real PyTorch training loop with AdamW, local epochs, and
  optional differential-privacy noise injection.
* **NVFlare** (``NVFlareRetinalClient``) -- minimal interface stub
  mapping NVFlare's ``Learner`` contract onto the same abstract base.

Two aggregation strategies are included:

* ``FedAvgStrategy`` -- weighted average of model parameters (McMahan
  et al., 2017).
* ``FedProxStrategy`` -- FedAvg with a proximal regularisation term
  that penalises local parameter drift (Li et al., 2020).

All functionality is gated behind ``FEDERATED__ENABLED=false`` by
default and Flower imports are guarded with ``try/except`` so the
module never breaks a deployment that lacks the ``flwr`` package.
"""
from __future__ import annotations

import copy
import logging
import sys
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# Project root for src imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Flower import
# ---------------------------------------------------------------------------

_flwr_available = False
try:
    import flwr as fl                                  # noqa: F401
    from flwr.client import NumPyClient                # noqa: F401
    from flwr.common import NDArrays                   # noqa: F401

    _flwr_available = True
except ImportError:
    logger.info(
        "Flower (flwr) not installed -- FlowerRetinalClient will be "
        "non-functional.  Install: pip install flwr"
    )

# ---------------------------------------------------------------------------
# Optional NVFlare import
# ---------------------------------------------------------------------------

_nvflare_available = False
try:
    import nvflare  # noqa: F401

    _nvflare_available = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class FederatedClient(ABC):
    """Abstract base class for a federated learning client.

    Every concrete client must expose methods to exchange model
    parameters with an aggregation server and to perform local
    training (``fit``) and evaluation (``evaluate``).
    """

    @abstractmethod
    def get_model_parameters(self) -> list[np.ndarray]:
        """Return the current model parameters as a list of NumPy arrays.

        The ordering must be deterministic and consistent across all
        clients participating in the same federation.
        """
        ...

    @abstractmethod
    def set_model_parameters(self, parameters: list[np.ndarray]) -> None:
        """Replace the model's parameters with the supplied arrays.

        Parameters
        ----------
        parameters : list[np.ndarray]
            Model parameters received from the aggregation server.
        """
        ...

    @abstractmethod
    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        """Perform local training on the client's dataset.

        Parameters
        ----------
        parameters : list[np.ndarray]
            Global model parameters to initialise from.
        config : dict
            Training configuration sent by the server (e.g. round
            number, local_epochs override).

        Returns
        -------
        tuple
            ``(updated_parameters, num_samples, metrics_dict)``
        """
        ...

    @abstractmethod
    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[float, int, dict[str, Any]]:
        """Evaluate the model on the client's validation set.

        Parameters
        ----------
        parameters : list[np.ndarray]
            Model parameters to evaluate.
        config : dict
            Evaluation configuration from the server.

        Returns
        -------
        tuple
            ``(loss, num_samples, metrics_dict)``
        """
        ...


# ---------------------------------------------------------------------------
# Flower client
# ---------------------------------------------------------------------------


class FlowerRetinalClient(FederatedClient):
    """Flower-compatible federated client wrapping a real PyTorch loop.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to train (e.g. ViGNN instance).
    train_loader : torch.utils.data.DataLoader
        Training data loader (local site data).
    val_loader : torch.utils.data.DataLoader
        Validation data loader.
    device : torch.device | None
        Compute device.  Defaults to CUDA if available.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        cfg = settings.federated
        self._local_epochs: int = cfg.local_epochs
        self._dp_enabled: bool = cfg.dp_enabled
        self._dp_epsilon: float = cfg.dp_epsilon

        logger.info(
            "FlowerRetinalClient created (local_epochs=%d, dp=%s, device=%s)",
            self._local_epochs,
            self._dp_enabled,
            self.device,
        )

    # -- FederatedClient interface -----------------------------------------

    def get_model_parameters(self) -> list[np.ndarray]:
        """Extract model parameters as a list of NumPy arrays."""
        return [
            val.cpu().numpy()
            for _, val in self.model.state_dict().items()
        ]

    def set_model_parameters(self, parameters: list[np.ndarray]) -> None:
        """Load parameters into the model."""
        state_dict = self.model.state_dict()
        params_dict = zip(state_dict.keys(), parameters)
        new_state = OrderedDict(
            {k: torch.tensor(v, dtype=state_dict[k].dtype) for k, v in params_dict}
        )
        self.model.load_state_dict(new_state, strict=True)

    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        """Train locally for ``local_epochs`` using AdamW.

        Parameters
        ----------
        parameters : list[np.ndarray]
            Global parameters to start from.
        config : dict
            May contain ``local_epochs`` override and ``round`` number.

        Returns
        -------
        tuple
            ``(updated_parameters, num_train_samples, {"loss": ..., "round": ...})``
        """
        self.set_model_parameters(parameters)
        epochs = config.get("local_epochs", self._local_epochs)
        current_round = config.get("round", 0)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=1e-4,
        )
        criterion = nn.BCEWithLogitsLoss()

        self.model.train()
        total_loss = 0.0
        num_samples = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                logits = self.model(images)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item() * images.size(0)
                num_samples += images.size(0)

            avg_epoch_loss = epoch_loss / max(len(self.train_loader.dataset), 1)
            total_loss = avg_epoch_loss
            logger.debug(
                "FL round %d, local epoch %d/%d, loss=%.4f",
                current_round,
                epoch + 1,
                epochs,
                avg_epoch_loss,
            )

        # Optional: add differential-privacy noise to gradients
        if self._dp_enabled:
            self._apply_dp_noise()

        updated_params = self.get_model_parameters()
        num_train = len(self.train_loader.dataset)

        return (
            updated_params,
            num_train,
            {"loss": total_loss, "round": current_round},
        )

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[float, int, dict[str, Any]]:
        """Evaluate the global model on the local validation set.

        Returns
        -------
        tuple
            ``(avg_loss, num_val_samples, {"accuracy": ..., "round": ...})``
        """
        self.set_model_parameters(parameters)
        criterion = nn.BCEWithLogitsLoss()

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)
                loss = criterion(logits, labels)
                total_loss += loss.item() * images.size(0)

                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.numel()

        num_val = len(self.val_loader.dataset)
        avg_loss = total_loss / max(num_val, 1)
        accuracy = correct / max(total, 1)
        current_round = config.get("round", 0)

        logger.info(
            "FL evaluation round %d: loss=%.4f accuracy=%.4f (%d samples)",
            current_round,
            avg_loss,
            accuracy,
            num_val,
        )

        return (
            avg_loss,
            num_val,
            {"accuracy": accuracy, "round": current_round},
        )

    # -- Differential privacy -----------------------------------------------

    def _apply_dp_noise(self) -> None:
        """Add calibrated Gaussian noise to model parameters for (epsilon, delta)-DP.

        Uses a simplified Gaussian mechanism. In production this should
        be replaced with a rigorous privacy accountant (e.g. Opacus or
        TensorFlow Privacy).
        """
        sensitivity = 1.0  # L2 sensitivity upper bound after gradient clipping
        delta = 1e-5
        sigma = (sensitivity * (2.0 * np.log(1.25 / delta)) ** 0.5) / self._dp_epsilon

        with torch.no_grad():
            for param in self.model.parameters():
                noise = torch.randn_like(param) * sigma
                param.add_(noise)

        logger.info(
            "DP noise applied (epsilon=%.2f, sigma=%.4f)",
            self._dp_epsilon,
            sigma,
        )

    # -- Flower NumPyClient adapter -----------------------------------------

    def to_flower_client(self) -> Any:
        """Return a Flower ``NumPyClient`` adapter wrapping this client.

        Returns ``None`` if Flower is not installed.
        """
        if not _flwr_available:
            logger.warning("Cannot create Flower client: flwr not installed")
            return None

        outer = self

        class _FlowerAdapter(NumPyClient):
            """Thin adapter mapping Flower's NumPyClient to our FederatedClient."""

            def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
                return outer.get_model_parameters()

            def fit(
                self,
                parameters: list[np.ndarray],
                config: dict[str, Any],
            ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
                return outer.fit(parameters, config)

            def evaluate(
                self,
                parameters: list[np.ndarray],
                config: dict[str, Any],
            ) -> tuple[float, int, dict[str, Any]]:
                return outer.evaluate(parameters, config)

        return _FlowerAdapter()


# ---------------------------------------------------------------------------
# NVFlare client (minimal stub)
# ---------------------------------------------------------------------------


class NVFlareRetinalClient(FederatedClient):
    """Minimal NVFlare interface stub.

    Maps NVFlare's ``Learner`` lifecycle methods onto the
    ``FederatedClient`` abstract base.  This is a skeleton; a full
    implementation would use ``nvflare.apis.fl_context.FLContext`` and
    the ``Shareable`` transport.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to train.
    train_loader : torch.utils.data.DataLoader
        Local training data loader.
    val_loader : torch.utils.data.DataLoader
        Local validation data loader.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        logger.info("NVFlareRetinalClient stub created")

    def get_model_parameters(self) -> list[np.ndarray]:
        return [
            val.cpu().numpy()
            for _, val in self.model.state_dict().items()
        ]

    def set_model_parameters(self, parameters: list[np.ndarray]) -> None:
        state_dict = self.model.state_dict()
        params_dict = zip(state_dict.keys(), parameters)
        new_state = OrderedDict(
            {k: torch.tensor(v, dtype=state_dict[k].dtype) for k, v in params_dict}
        )
        self.model.load_state_dict(new_state, strict=True)

    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        """Placeholder local training for NVFlare.

        Delegates to a simple training loop identical in structure to
        the Flower client's, but without the DP noise layer.
        """
        self.set_model_parameters(parameters)
        epochs = config.get("local_epochs", settings.federated.local_epochs)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        self.model.train()
        total_loss = 0.0

        for epoch in range(epochs):
            epoch_loss = 0.0
            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                logits = self.model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * images.size(0)

            total_loss = epoch_loss / max(len(self.train_loader.dataset), 1)

        return (
            self.get_model_parameters(),
            len(self.train_loader.dataset),
            {"loss": total_loss},
        )

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[float, int, dict[str, Any]]:
        """Placeholder evaluation for NVFlare."""
        self.set_model_parameters(parameters)
        criterion = nn.BCEWithLogitsLoss()

        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(images)
                loss = criterion(logits, labels)
                total_loss += loss.item() * images.size(0)

        num_val = len(self.val_loader.dataset)
        avg_loss = total_loss / max(num_val, 1)
        return (avg_loss, num_val, {"loss": avg_loss})


# ---------------------------------------------------------------------------
# Aggregation strategies
# ---------------------------------------------------------------------------


class FedAvgStrategy:
    """Federated Averaging: weighted average of model parameters.

    Implements the FedAvg algorithm (McMahan et al., 2017).  Weights
    are proportional to each client's number of training samples.
    """

    @staticmethod
    def aggregate(
        results: list[tuple[list[np.ndarray], int]],
    ) -> list[np.ndarray]:
        """Aggregate model parameters from multiple clients.

        Parameters
        ----------
        results : list[tuple[list[np.ndarray], int]]
            Each entry is ``(client_parameters, num_samples)``.

        Returns
        -------
        list[np.ndarray]
            Weighted-average parameters.
        """
        total_samples = sum(n for _, n in results)
        if total_samples == 0:
            logger.warning("FedAvg: total samples is 0, returning first client params")
            return results[0][0] if results else []

        # Initialise with zeros
        num_layers = len(results[0][0])
        aggregated: list[np.ndarray] = [
            np.zeros_like(results[0][0][i]) for i in range(num_layers)
        ]

        for params, num_samples in results:
            weight = num_samples / total_samples
            for i in range(num_layers):
                aggregated[i] += params[i] * weight

        logger.info(
            "FedAvg aggregated %d clients (%d total samples)",
            len(results),
            total_samples,
        )
        return aggregated


class FedProxStrategy:
    """Federated Proximal: FedAvg with proximal regularisation.

    Adds a proximal term ``(mu / 2) * ||w - w_global||^2`` to each
    client's local objective, penalising deviation from the global
    model (Li et al., 2020).

    The aggregation step is identical to FedAvg; the proximal penalty
    is applied during local training by modifying the loss function.

    Parameters
    ----------
    mu : float
        Proximal regularisation strength.  Higher values keep clients
        closer to the global model.
    """

    def __init__(self, mu: float = 0.01) -> None:
        self.mu = mu

    def compute_proximal_term(
        self,
        model: nn.Module,
        global_params: list[np.ndarray],
    ) -> torch.Tensor:
        """Compute the proximal penalty ``(mu / 2) * ||w - w_global||^2``.

        Parameters
        ----------
        model : nn.Module
            The local model whose parameters are being updated.
        global_params : list[np.ndarray]
            The global model parameters from the latest round.

        Returns
        -------
        torch.Tensor
            Scalar proximal penalty term to add to the local loss.
        """
        device = next(model.parameters()).device
        proximal_term = torch.tensor(0.0, device=device)

        for local_param, global_np in zip(model.parameters(), global_params):
            global_tensor = torch.tensor(
                global_np, dtype=local_param.dtype, device=device
            )
            proximal_term += ((local_param - global_tensor) ** 2).sum()

        return (self.mu / 2.0) * proximal_term

    def aggregate(
        self,
        results: list[tuple[list[np.ndarray], int]],
    ) -> list[np.ndarray]:
        """Aggregate using standard FedAvg weighted average.

        The proximal difference is in the local training objective, not
        in the aggregation step.
        """
        return FedAvgStrategy.aggregate(results)


# ---------------------------------------------------------------------------
# Factory / entrypoint
# ---------------------------------------------------------------------------


def create_federated_client(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
) -> FederatedClient | None:
    """Create a federated client based on ``settings.federated``.

    Returns ``None`` when federated learning is disabled.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to train.
    train_loader, val_loader : DataLoader
        Local data loaders for training and validation.
    """
    cfg = settings.federated
    if not cfg.enabled:
        logger.info("Federated learning disabled (FEDERATED__ENABLED=false)")
        return None

    framework = cfg.framework.lower()

    if framework == "flower":
        client = FlowerRetinalClient(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
        )
        logger.info("Flower federated client created")
        return client
    elif framework == "nvflare":
        client = NVFlareRetinalClient(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
        )
        logger.info("NVFlare federated client stub created")
        return client
    else:
        logger.error("Unknown federated framework: %s", framework)
        return None


def start_flower_client(
    client: FlowerRetinalClient,
    server_address: str | None = None,
) -> None:
    """Connect a FlowerRetinalClient to a Flower server and start training.

    This is a blocking call that runs until the server completes all
    federated rounds.

    Parameters
    ----------
    client : FlowerRetinalClient
        The client to connect.
    server_address : str | None
        Flower server address.  Defaults to ``settings.federated.server_address``.
    """
    if not _flwr_available:
        logger.error("Cannot start Flower client: flwr package not installed")
        return

    address = server_address or settings.federated.server_address
    flower_client = client.to_flower_client()

    if flower_client is None:
        logger.error("Failed to create Flower NumPyClient adapter")
        return

    logger.info("Starting Flower client, connecting to %s", address)
    fl.client.start_client(
        server_address=address,
        client=flower_client,
    )
