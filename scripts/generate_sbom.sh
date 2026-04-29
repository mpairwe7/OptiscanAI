#!/usr/bin/env bash
# Generate SBOM and scan vulnerabilities for RetinalAI Docker image.
#
# Produces:
#   outputs/sbom/<image>-sbom.spdx.json    SPDX JSON SBOM (via Syft)
#   outputs/sbom/<image>-vulns.json        Vulnerability report (via Grype)
#   outputs/sbom/python-sbom.spdx.json     Python-specific SBOM from pyproject.toml
#
# Usage:
#   ./scripts/generate_sbom.sh                        # default: retinal-ai:latest
#   ./scripts/generate_sbom.sh myregistry/app:v2.1.0
#
# Exit codes:
#   0  No critical vulnerabilities found
#   1  Critical vulnerabilities detected or tool missing
#
# Requirements:
#   - syft  (https://github.com/anchore/syft)
#   - grype (https://github.com/anchore/grype)

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

IMAGE="${1:-retinal-ai:latest}"
# Sanitise image name for use in filenames (replace / and : with -)
SAFE_NAME="${IMAGE//\//-}"
SAFE_NAME="${SAFE_NAME//:/-}"

OUTPUT_DIR="outputs/sbom"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

# ── Colour helpers ────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No colour

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }

# ── Pre-flight checks ────────────────────────────────────────────────────────

check_tool() {
    if ! command -v "$1" &>/dev/null; then
        error "'$1' is not installed.  Install it from: $2"
        exit 1
    fi
}

check_tool syft  "https://github.com/anchore/syft#installation"
check_tool grype "https://github.com/anchore/grype#installation"

# ── Create output directory ──────────────────────────────────────────────────

mkdir -p "${OUTPUT_DIR}"
info "Output directory: ${OUTPUT_DIR}"

# ── Step 1: Generate SPDX JSON SBOM from Docker image ────────────────────────

SBOM_FILE="${OUTPUT_DIR}/${SAFE_NAME}-sbom.spdx.json"

info "Generating SPDX SBOM for image '${IMAGE}' ..."
syft "${IMAGE}" \
    --output spdx-json="${SBOM_FILE}" \
    --quiet

if [[ -f "${SBOM_FILE}" ]]; then
    SBOM_SIZE=$(du -h "${SBOM_FILE}" | cut -f1)
    ok "SBOM written to ${SBOM_FILE} (${SBOM_SIZE})"
else
    error "SBOM generation failed — file not created."
    exit 1
fi

# ── Step 2: Scan for vulnerabilities with Grype ──────────────────────────────

VULN_FILE="${OUTPUT_DIR}/${SAFE_NAME}-vulns.json"

info "Scanning for vulnerabilities ..."
grype "sbom:${SBOM_FILE}" \
    --output json \
    --file "${VULN_FILE}" \
    --quiet || true  # grype returns non-zero on findings; we handle below

if [[ ! -f "${VULN_FILE}" ]]; then
    error "Vulnerability scan failed — report not created."
    exit 1
fi

ok "Vulnerability report written to ${VULN_FILE}"

# ── Step 3: Python-specific SBOM from pyproject.toml ─────────────────────────

PYPROJECT="pyproject.toml"
PY_SBOM_FILE="${OUTPUT_DIR}/python-sbom.spdx.json"

if [[ -f "${PYPROJECT}" ]]; then
    info "Generating Python-specific SBOM from ${PYPROJECT} ..."
    syft "dir:." \
        --source-name "retinalai-python-deps" \
        --output spdx-json="${PY_SBOM_FILE}" \
        --quiet

    if [[ -f "${PY_SBOM_FILE}" ]]; then
        ok "Python SBOM written to ${PY_SBOM_FILE}"
    else
        warn "Python SBOM generation did not produce output."
    fi
else
    warn "${PYPROJECT} not found; skipping Python-specific SBOM."
fi

# ── Step 4: Summary ──────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "  RetinalAI SBOM Summary  —  ${TIMESTAMP}"
echo "============================================================"
echo "  Image:             ${IMAGE}"
echo "  SBOM (SPDX):       ${SBOM_FILE}"
echo "  Vuln report:       ${VULN_FILE}"
if [[ -f "${PY_SBOM_FILE}" ]]; then
    echo "  Python SBOM:       ${PY_SBOM_FILE}"
fi

# Parse vulnerability counts from the JSON report
CRITICAL=0
HIGH=0
MEDIUM=0
LOW=0

if command -v jq &>/dev/null && [[ -f "${VULN_FILE}" ]]; then
    CRITICAL=$(jq '[.matches[]? | select(.vulnerability.severity == "Critical")] | length' "${VULN_FILE}" 2>/dev/null || echo 0)
    HIGH=$(jq '[.matches[]? | select(.vulnerability.severity == "High")] | length' "${VULN_FILE}" 2>/dev/null || echo 0)
    MEDIUM=$(jq '[.matches[]? | select(.vulnerability.severity == "Medium")] | length' "${VULN_FILE}" 2>/dev/null || echo 0)
    LOW=$(jq '[.matches[]? | select(.vulnerability.severity == "Low")] | length' "${VULN_FILE}" 2>/dev/null || echo 0)
elif command -v python3 &>/dev/null && [[ -f "${VULN_FILE}" ]]; then
    # Fallback: use Python if jq is not available
    read -r CRITICAL HIGH MEDIUM LOW <<< "$(python3 -c "
import json, sys
with open('${VULN_FILE}') as f:
    data = json.load(f)
matches = data.get('matches', [])
counts = {'Critical': 0, 'High': 0, 'Medium': 0, 'Low': 0}
for m in matches:
    sev = m.get('vulnerability', {}).get('severity', '')
    if sev in counts:
        counts[sev] += 1
print(counts['Critical'], counts['High'], counts['Medium'], counts['Low'])
" 2>/dev/null || echo "0 0 0 0")"
fi

echo ""
echo "  Vulnerabilities:"
echo "    Critical:  ${CRITICAL}"
echo "    High:      ${HIGH}"
echo "    Medium:    ${MEDIUM}"
echo "    Low:       ${LOW}"
echo "============================================================"
echo ""

# ── Step 5: Fail on critical vulnerabilities ─────────────────────────────────

if [[ "${CRITICAL}" -gt 0 ]]; then
    error "${CRITICAL} CRITICAL vulnerability(ies) found!  Review ${VULN_FILE} and remediate before release."
    exit 1
fi

ok "No critical vulnerabilities found.  SBOM artifacts are ready for release."
exit 0
