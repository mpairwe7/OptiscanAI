#!/usr/bin/env python3
"""Fine-tune Whisper-tiny for Ugandan English + Luganda code-switching.

Usage:
    PYTHONPATH=. python scripts/finetune_whisper_uganda.py \
        --data-dir data/voice/uganda_asr \
        --output-dir models/voice \
        --epochs 10

Expects data directory structure:
    data/voice/uganda_asr/
        train/
            *.wav + *.txt pairs
        test/
            *.wav + *.txt pairs

Produces:
    models/voice/
        whisper-tiny-ug/         # HuggingFace model directory
        whisper-tiny-ug.onnx     # ONNX export for on-device use
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def prepare_dataset(data_dir: Path, split: str = "train"):
    """Load wav+txt pairs into a HuggingFace Dataset."""
    from datasets import Audio, Dataset

    split_dir = data_dir / split
    if not split_dir.exists():
        logger.warning("Split directory not found: %s", split_dir)
        return None

    records = []
    for wav_file in sorted(split_dir.glob("*.wav")):
        txt_file = wav_file.with_suffix(".txt")
        if txt_file.exists():
            records.append({
                "audio": str(wav_file),
                "text": txt_file.read_text().strip(),
            })

    if not records:
        logger.warning("No audio+text pairs found in %s", split_dir)
        return None

    ds = Dataset.from_list(records)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    logger.info("Loaded %d samples from %s/%s", len(records), data_dir, split)
    return ds


def finetune(args):
    """Run Whisper-tiny fine-tuning."""
    import torch
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) / "whisper-tiny-ug"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and processor
    model_name = "openai/whisper-tiny"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # Prepare datasets
    train_ds = prepare_dataset(data_dir, "train")
    test_ds = prepare_dataset(data_dir, "test")

    if train_ds is None:
        logger.error("No training data found. Creating example structure.")
        example_dir = data_dir / "train"
        example_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "README.md").write_text(
            "# Uganda ASR Training Data\n\n"
            "Place WAV files and corresponding TXT transcription files here.\n"
            "Example: patient_001.wav + patient_001.txt\n\n"
            "Minimum 200 utterances for the test set (WER evaluation).\n"
            "Target: WER <= 18% on Ugandan English + Luganda code-switching.\n"
        )
        logger.info("Created data directory template at %s", data_dir)
        return

    def prepare_features(batch):
        audio = batch["audio"]
        input_features = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt",
        ).input_features[0]

        labels = processor.tokenizer(batch["text"]).input_ids
        return {"input_features": input_features, "labels": labels}

    train_ds = train_ds.map(prepare_features, remove_columns=train_ds.column_names)
    if test_ds:
        test_ds = test_ds.map(prepare_features, remove_columns=test_ds.column_names)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        learning_rate=1e-5,
        warmup_steps=100,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch" if test_ds else "no",
        save_strategy="epoch",
        save_total_limit=3,
        logging_steps=25,
        predict_with_generate=True,
        generation_max_length=225,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=processor,
    )

    logger.info("Starting fine-tuning: %d epochs, %d samples", args.epochs, len(train_ds))
    trainer.train()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    logger.info("Model saved to %s", output_dir)

    # Export to ONNX (optional, requires optimum)
    try:
        from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

        onnx_path = Path(args.output_dir) / "whisper-tiny-ug.onnx"
        ort_model = ORTModelForSpeechSeq2Seq.from_pretrained(
            str(output_dir), export=True
        )
        ort_model.save_pretrained(str(onnx_path.parent / "whisper-tiny-ug-onnx"))
        logger.info("ONNX export saved")
    except ImportError:
        logger.info("optimum not installed — skipping ONNX export")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper for Uganda")
    parser.add_argument("--data-dir", type=str, default="data/voice/uganda_asr")
    parser.add_argument("--output-dir", type=str, default="models/voice")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    finetune(args)


if __name__ == "__main__":
    main()
