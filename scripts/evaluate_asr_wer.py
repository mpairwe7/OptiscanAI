#!/usr/bin/env python3
"""Evaluate ASR Word Error Rate on Uganda-specific test set.

Usage:
    PYTHONPATH=. python scripts/evaluate_asr_wer.py \
        --model-path models/voice/whisper-tiny-ug \
        --test-dir data/voice/uganda_asr/test \
        --output outputs/asr_evaluation.json

Reports:
    - Overall WER
    - WER by language (English-only, Luganda-only, code-switched)
    - Per-clinical-term accuracy
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate between reference and hypothesis."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    # Levenshtein distance at word level
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,  # deletion
                d[i][j - 1] + 1,  # insertion
                d[i - 1][j - 1] + cost,  # substitution
            )

    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def classify_language(text: str) -> str:
    """Classify text as English, Luganda, or code-switched."""
    luganda_prefixes = ("oku", "obu", "emu", "aba", "ama", "eby", "eki", "omu")
    words = text.lower().split()
    if not words:
        return "unknown"

    lg_count = sum(1 for w in words if any(w.startswith(p) for p in luganda_prefixes))
    lg_ratio = lg_count / len(words)

    if lg_ratio > 0.6:
        return "luganda"
    elif lg_ratio > 0.2:
        return "code_switched"
    return "english"


def evaluate(args):
    """Run WER evaluation on test set."""
    test_dir = Path(args.test_dir)

    # Collect test pairs
    pairs = []
    for wav_file in sorted(test_dir.glob("*.wav")):
        txt_file = wav_file.with_suffix(".txt")
        if txt_file.exists():
            pairs.append(
                {
                    "audio": str(wav_file),
                    "reference": txt_file.read_text().strip(),
                }
            )

    if not pairs:
        logger.error("No test pairs found in %s", test_dir)
        print(f"No test data found. Place WAV+TXT pairs in {test_dir}")
        return

    logger.info("Evaluating %d test utterances", len(pairs))

    # Load model
    try:
        from faster_whisper import WhisperModel

        model = WhisperModel(args.model_path, device="cpu", compute_type="int8")
    except ImportError:
        logger.error("faster-whisper not installed")
        return

    # Run evaluation
    results_by_lang = {"english": [], "luganda": [], "code_switched": [], "unknown": []}
    all_wers = []

    clinical_terms = [
        "diabetes",
        "sugar",
        "hypertension",
        "pressure",
        "hiv",
        "sickle",
        "malaria",
        "esukaali",
        "puleesa",
        "silimu",
        "amaaso",
    ]
    term_correct = {t: 0 for t in clinical_terms}
    term_total = {t: 0 for t in clinical_terms}

    for pair in pairs:
        ref = pair["reference"]
        lang = classify_language(ref)

        # Transcribe
        import soundfile as sf

        audio, sr = sf.read(pair["audio"])
        if sr != 16000:
            import resampy

            audio = resampy.resample(audio, sr, 16000)

        segments, _ = model.transcribe(audio, language="en", beam_size=5)
        hyp = " ".join(seg.text.strip() for seg in segments)

        wer = compute_wer(ref, hyp)
        all_wers.append(wer)
        results_by_lang[lang].append(wer)

        # Clinical term accuracy
        ref_lower = ref.lower()
        hyp_lower = hyp.lower()
        for term in clinical_terms:
            if term in ref_lower:
                term_total[term] += 1
                if term in hyp_lower:
                    term_correct[term] += 1

    # Compute metrics
    overall_wer = np.mean(all_wers) if all_wers else 0.0
    report = {
        "total_utterances": len(pairs),
        "overall_wer": round(float(overall_wer * 100), 2),
        "wer_target": 18.0,
        "target_met": overall_wer * 100 <= 18.0,
        "by_language": {},
        "clinical_term_accuracy": {},
    }

    for lang, wers in results_by_lang.items():
        if wers:
            report["by_language"][lang] = {
                "count": len(wers),
                "wer": round(float(np.mean(wers) * 100), 2),
            }

    for term in clinical_terms:
        if term_total[term] > 0:
            acc = term_correct[term] / term_total[term]
            report["clinical_term_accuracy"][term] = {
                "correct": term_correct[term],
                "total": term_total[term],
                "accuracy": round(acc * 100, 1),
            }

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nASR Evaluation {'PASSED' if report['target_met'] else 'FAILED'}")
    print(f"  Overall WER: {report['overall_wer']:.1f}% (target: <= {report['wer_target']}%)")
    for lang, data in report["by_language"].items():
        print(f"  {lang}: WER={data['wer']:.1f}% ({data['count']} utterances)")
    print(f"  Report: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Uganda ASR WER")
    parser.add_argument("--model-path", type=str, default="models/voice/whisper-tiny-ug")
    parser.add_argument("--test-dir", type=str, default="data/voice/uganda_asr/test")
    parser.add_argument("--output", type=str, default="outputs/asr_evaluation.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    evaluate(args)


if __name__ == "__main__":
    main()
