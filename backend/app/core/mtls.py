"""
Mutual TLS (mTLS) configuration for inter-service communication.

Provides certificate-based authentication between RetinalAI micro-services
(API gateway <-> Ray Serve, API <-> Kafka, API <-> MLflow).

Features:
    - SSLContext construction for both server and client sides
    - httpx-compatible verification output for async HTTP clients
    - Self-signed certificate generation for development/testing
    - Settings-driven configuration via ``backend.app.core.config``

Security note:
    In production, certificates MUST be issued by a private CA (e.g. Vault PKI,
    AWS Private CA, or step-ca).  The ``create_dev_certificates`` helper is
    strictly for local development and CI.
"""

from __future__ import annotations

import logging
import os
import ssl
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Union

logger = logging.getLogger(__name__)


class MTLSConfig:
    """Mutual TLS configuration sourced from application settings.

    Parameters
    ----------
    enabled : bool
        Whether mTLS is active.
    ca_cert_path : str
        Path to the CA certificate bundle.
    client_cert_path : str
        Path to the client certificate (PEM).
    client_key_path : str
        Path to the client private key (PEM).
    verify_hostname : bool
        Whether to verify the server hostname against the certificate.
    """

    def __init__(
        self,
        enabled: bool = False,
        ca_cert_path: str = "",
        client_cert_path: str = "",
        client_key_path: str = "",
        verify_hostname: bool = True,
    ) -> None:
        self.enabled = enabled
        self.ca_cert_path = ca_cert_path
        self.client_cert_path = client_cert_path
        self.client_key_path = client_key_path
        self.verify_hostname = verify_hostname

    # ------------------------------------------------------------------
    # SSL context construction
    # ------------------------------------------------------------------

    def to_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Create an ``ssl.SSLContext`` for server or client connections.

        Returns ``None`` when mTLS is disabled so callers can skip TLS
        configuration entirely.

        Returns
        -------
        ssl.SSLContext | None
            Configured context or ``None`` if mTLS is disabled.

        Raises
        ------
        FileNotFoundError
            If any referenced certificate file does not exist.
        ssl.SSLError
            If the certificates or key cannot be loaded.
        """
        if not self.enabled:
            logger.debug("mTLS disabled; returning None SSLContext")
            return None

        self._validate_paths()

        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

        # Require client certificates from the peer
        ctx.verify_mode = ssl.CERT_REQUIRED
        ctx.check_hostname = self.verify_hostname

        # Load trusted CA
        ctx.load_verify_locations(cafile=self.ca_cert_path)
        logger.debug("Loaded CA certificate: %s", self.ca_cert_path)

        # Load client cert + key for mutual authentication
        if self.client_cert_path and self.client_key_path:
            ctx.load_cert_chain(
                certfile=self.client_cert_path,
                keyfile=self.client_key_path,
            )
            logger.debug(
                "Loaded client cert: %s, key: %s",
                self.client_cert_path,
                self.client_key_path,
            )

        # Restrict to TLS 1.2+
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2

        logger.info("mTLS SSLContext created (verify_hostname=%s)", self.verify_hostname)
        return ctx

    # ------------------------------------------------------------------
    # httpx integration
    # ------------------------------------------------------------------

    def to_httpx_verify(self) -> Union[str, bool, ssl.SSLContext]:
        """Return a value suitable for ``httpx.AsyncClient(verify=...)``.

        * mTLS disabled and no CA path -> ``True`` (default system CAs)
        * mTLS disabled but CA path set -> CA path string
        * mTLS enabled -> full ``ssl.SSLContext`` with client cert

        Returns
        -------
        str | bool | ssl.SSLContext
        """
        if not self.enabled:
            if self.ca_cert_path and os.path.isfile(self.ca_cert_path):
                return self.ca_cert_path
            return True

        ctx = self.to_ssl_context()
        if ctx is not None:
            return ctx

        # Fallback (should not normally be reached)
        return True

    # ------------------------------------------------------------------
    # Development certificate generation
    # ------------------------------------------------------------------

    @staticmethod
    def create_dev_certificates(output_dir: str = "certs/") -> Dict[str, str]:
        """Generate self-signed CA, server, and client certificates.

        Uses the ``cryptography`` library to produce a minimal PKI suitable
        for local development and CI testing.  **Never use these certificates
        in production.**

        Parameters
        ----------
        output_dir : str
            Directory to write the generated PEM files.

        Returns
        -------
        dict
            Mapping of ``{ca_cert, ca_key, server_cert, server_key,
            client_cert, client_key}`` to their absolute file paths.

        Raises
        ------
        ImportError
            If the ``cryptography`` library is not installed.
        """
        try:
            from cryptography import x509
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import rsa
            from cryptography.x509.oid import NameOID
        except ImportError:
            raise ImportError(
                "The 'cryptography' package is required for certificate "
                "generation.  Install it with:  pip install cryptography"
            )

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc)
        validity = timedelta(days=365)

        # ---- CA key + certificate ----
        ca_key = rsa.generate_private_key(public_exponent=65537, key_size=4096)
        ca_name = x509.Name(
            [
                x509.NameAttribute(NameOID.COUNTRY_NAME, "UG"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "RetinalAI Dev CA"),
                x509.NameAttribute(NameOID.COMMON_NAME, "RetinalAI Development CA"),
            ]
        )
        ca_cert = (
            x509.CertificateBuilder()
            .subject_name(ca_name)
            .issuer_name(ca_name)
            .public_key(ca_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + validity)
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    content_commitment=False,
                    key_encipherment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=True,
                    crl_sign=True,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .sign(ca_key, hashes.SHA256())
        )

        def _write_key(key: rsa.RSAPrivateKey, path: Path) -> None:
            path.write_bytes(
                key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )
            # Restrict permissions
            path.chmod(0o600)

        def _write_cert(cert: x509.Certificate, path: Path) -> None:
            path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))

        def _issue_cert(
            common_name: str,
            san_dns: list[str],
            is_server: bool,
        ) -> tuple[rsa.RSAPrivateKey, x509.Certificate]:
            key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            subject = x509.Name(
                [
                    x509.NameAttribute(NameOID.COUNTRY_NAME, "UG"),
                    x509.NameAttribute(NameOID.ORGANIZATION_NAME, "RetinalAI"),
                    x509.NameAttribute(NameOID.COMMON_NAME, common_name),
                ]
            )
            usage_ext = x509.ExtendedKeyUsage(
                [
                    (
                        x509.oid.ExtendedKeyUsageOID.SERVER_AUTH
                        if is_server
                        else x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH
                    )
                ]
            )
            san = x509.SubjectAlternativeName([x509.DNSName(d) for d in san_dns])
            cert = (
                x509.CertificateBuilder()
                .subject_name(subject)
                .issuer_name(ca_name)
                .public_key(key.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(now)
                .not_valid_after(now + validity)
                .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
                .add_extension(usage_ext, critical=False)
                .add_extension(san, critical=False)
                .sign(ca_key, hashes.SHA256())
            )
            return key, cert

        # ---- Server certificate ----
        server_key, server_cert = _issue_cert(
            common_name="retinalai-server",
            san_dns=["localhost", "retinalai-api", "*.retinalai.local"],
            is_server=True,
        )

        # ---- Client certificate ----
        client_key, client_cert = _issue_cert(
            common_name="retinalai-client",
            san_dns=["retinalai-client"],
            is_server=False,
        )

        # ---- Write files ----
        paths: Dict[str, str] = {}

        for name, key_obj, cert_obj in [
            ("ca", ca_key, ca_cert),
            ("server", server_key, server_cert),
            ("client", client_key, client_cert),
        ]:
            key_path = out / f"{name}.key"
            cert_path = out / f"{name}.crt"
            _write_key(key_obj, key_path)
            _write_cert(cert_obj, cert_path)
            paths[f"{name}_key"] = str(key_path.resolve())
            paths[f"{name}_cert"] = str(cert_path.resolve())

        logger.info(
            "Development certificates generated in %s (CA + server + client)",
            out.resolve(),
        )
        return paths

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_paths(self) -> None:
        """Ensure all configured certificate paths exist on disk."""
        for label, path in [
            ("CA certificate", self.ca_cert_path),
            ("client certificate", self.client_cert_path),
            ("client key", self.client_key_path),
        ]:
            if path and not os.path.isfile(path):
                raise FileNotFoundError(
                    f"mTLS {label} not found: {path}.  "
                    f"Set MTLS__ENABLED=false or provide valid paths."
                )

    def __repr__(self) -> str:
        return (
            f"MTLSConfig(enabled={self.enabled}, "
            f"ca={self.ca_cert_path!r}, "
            f"cert={self.client_cert_path!r}, "
            f"verify_hostname={self.verify_hostname})"
        )


# -----------------------------------------------------------------------
# Module-level loader
# -----------------------------------------------------------------------


def load_mtls_config() -> MTLSConfig:
    """Load mTLS configuration from application settings.

    Reads ``settings.mtls`` (an ``MTLSSettings`` instance) and returns a
    fully initialised ``MTLSConfig``.

    Returns
    -------
    MTLSConfig
    """
    from backend.app.core.config import settings

    mtls_settings = settings.mtls
    config = MTLSConfig(
        enabled=mtls_settings.enabled,
        ca_cert_path=mtls_settings.ca_cert_path,
        client_cert_path=mtls_settings.client_cert_path,
        client_key_path=mtls_settings.client_key_path,
        verify_hostname=mtls_settings.verify_hostname,
    )
    logger.info("mTLS config loaded: %s", config)
    return config
