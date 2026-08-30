#!/usr/bin/env python3
"""Export the fitted triage head to pickle-free JSON weights.

The selected triage model (``feat_logreg``) is a plain multinomial logistic
regression over 27 structured features. Shipping it as a joblib pickle would
mean the production image must (a) keep a scikit-learn version that can
unpickle it and (b) execute a pickle at startup — a code-execution surface for
a model that is, mathematically, one matrix multiply.

This exports the decision function to JSON instead::

    priority_index = classes[argmax(x @ coef.T + intercept)]

and **verifies that the exported form reproduces ``sklearn.predict`` exactly on
every available case** before writing anything. If a single case disagrees the
export aborts, so train/serve skew cannot ship silently.

    PYTHONPATH=. python3 scripts/export_triage_model.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.reasoner_comparison.cases import DISEASE_VOCAB  # noqa: E402
from src.evaluation.reasoner_comparison.features import (  # noqa: E402
    case_features,
    feature_names,
)
from src.evaluation.reasoner_comparison.interface import (  # noqa: E402
    PRIORITIES,
    Case,
    Prediction,
)

logger = logging.getLogger("export_triage")


def load_cases(paths: list[Path]) -> list[Case]:
    """Every trace we have, for the equivalence check (train + out-of-sample)."""
    cases: list[Case] = []
    for path in paths:
        if not path.exists():
            logger.warning("skipping missing %s", path)
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            preds = [
                Prediction(
                    p["code"], p["name"], float(p["probability"]), p.get("confidence", "unknown")
                )
                for p in row["predictions"]
            ]
            cases.append(
                Case(
                    scan_id=str(row.get("scan_id", "")),
                    predictions=preds,
                    probabilities={},
                    referral_priority=row.get("referral", "FOLLOW_UP"),
                )
            )
    return cases


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="outputs/triage_model/triage_model.joblib")
    p.add_argument("--out", default="models/triage/triage_model.json")
    p.add_argument(
        "--cases",
        nargs="*",
        default=[
            "outputs/reasoner_comparison_real/sft_data.jsonl",
            "outputs/generalizability/oos_traces.jsonl",
        ],
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    import joblib
    import numpy as np

    wrapper = joblib.load(args.model)
    base = wrapper.base
    if hasattr(base, "steps"):  # tolerate a Pipeline if the recipe ever adds one
        raise SystemExit(
            "fitted model is a Pipeline; this exporter only handles a bare "
            "LogisticRegression (no preprocessing to reproduce)"
        )
    if type(base).__name__ != "LogisticRegression":
        raise SystemExit(f"expected LogisticRegression, got {type(base).__name__}")

    coef = np.asarray(base.coef_, dtype=float)
    intercept = np.asarray(base.intercept_, dtype=float)
    classes = [int(c) for c in wrapper.classes_]
    names = feature_names(include_referral=True)
    if coef.shape[1] != len(names):
        raise SystemExit(f"feature count mismatch: coef has {coef.shape[1]}, names {len(names)}")

    cases = load_cases([Path(c) for c in args.cases])
    if not cases:
        raise SystemExit("no cases found — cannot verify the export")
    X = np.asarray([case_features(c, include_referral=True) for c in cases], dtype=float)

    # The whole point of the export: prove the JSON forward pass is the model.
    sk = wrapper.predict(X)
    exported = np.asarray(classes)[np.argmax(X @ coef.T + intercept, axis=1)]
    disagreements = int((sk != exported).sum())
    logger.info("equivalence check over %d cases: %d disagreements", len(cases), disagreements)
    if disagreements:
        raise SystemExit(
            f"ABORT: exported weights disagree with sklearn on {disagreements}/{len(cases)} "
            "cases — refusing to write a model that would silently drift at serve time"
        )

    payload = {
        "format": "linear-softmax-v1",
        "note": (
            "priority_index = classes[argmax(x @ coef.T + intercept)]; features in "
            "feature_names order. Verified identical to the fitted sklearn model on "
            f"{len(cases)} cases at export time."
        ),
        "priorities": list(PRIORITIES),
        "classes": classes,
        # The artifact is self-describing: the serving encoder builds its vector
        # from *these* columns and asserts the derived names match, so a change to
        # the disease vocabulary can never silently reorder features at serve time.
        "disease_codes": [code for code, _ in DISEASE_VOCAB],
        "feature_names": names,
        "coef": [[float(v) for v in row] for row in coef],
        "intercept": [float(v) for v in intercept],
        "trained_classes_note": (
            "EMERGENCY was absent from the training sample, so the learned head can "
            "never emit it. Emergency escalation is enforced deterministically by an "
            "override in src/triage/model.py, not by this model."
        ),
        "verified_on_cases": len(cases),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("wrote %s (%.1f KB)", out, out.stat().st_size / 1024)


if __name__ == "__main__":
    main()
