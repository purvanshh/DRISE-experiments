"""LayoutLMv3 fine-tuning on CORD.

Provides a complete training loop with:
- LayoutLMv3ForTokenClassification from HuggingFace transformers
- AdamW optimizer with linear warmup scheduler
- Per-epoch validation using seqeval entity-level F1
- Checkpoint saving (best model by val F1)
- CLI entrypoint: python -m document_intelligence_engine.multimodal.training
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from seqeval.metrics import f1_score as seqeval_f1
from seqeval.metrics import classification_report
from transformers import (
    LayoutLMv3ForTokenClassification,
    get_linear_schedule_with_warmup,
)

from document_intelligence_engine.multimodal.cord_dataset import (
    ID2LABEL,
    LABEL2ID,
    LABEL_LIST,
    NUM_LABELS,
    get_cord_dataloaders,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)


class LayoutLMv3Trainer:
    """Fine-tuning wrapper for LayoutLMv3 token classification on CORD."""

    def __init__(
        self,
        model_name: str = "microsoft/layoutlmv3-base",
        num_epochs: int = 10,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        gradient_accumulation_steps: int = 2,
        batch_size: int = 4,
        max_length: int = 512,
        max_train_samples: int | None = None,
        save_dir: str = "experiments/artifacts/cord_finetuned",
        eval_every_n_epochs: int = 1,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.batch_size = batch_size
        self.max_length = max_length
        self.max_train_samples = max_train_samples
        self.save_dir = Path(save_dir)
        self.eval_every_n_epochs = eval_every_n_epochs

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._model: LayoutLMv3ForTokenClassification | None = None
        self._best_f1: float = 0.0

    def train(self) -> dict[str, Any]:
        """Run the full training pipeline. Returns summary metrics."""
        logger.info("=== LayoutLMv3 Training on CORD ===")
        logger.info("Device: %s", self.device)
        logger.info("Epochs: %d, LR: %s, Batch: %d, Accum: %d",
                     self.num_epochs, self.learning_rate, self.batch_size,
                     self.gradient_accumulation_steps)

        # 1. Prepare data
        logger.info("Loading CORD dataset...")
        train_loader, val_loader, label_list = get_cord_dataloaders(
            model_name=self.model_name,
            batch_size=self.batch_size,
            max_length=self.max_length,
            max_train_samples=self.max_train_samples,
        )
        logger.info("Train batches: %d, Val batches: %d", len(train_loader), len(val_loader))

        # 2. Initialize model
        logger.info("Loading model: %s", self.model_name)
        self._model = LayoutLMv3ForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        self._model.to(self.device)

        # 3. Optimizer and scheduler
        no_decay = {"bias", "LayerNorm.weight"}
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self._model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self._model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate)

        total_steps = (len(train_loader) // self.gradient_accumulation_steps) * self.num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
        )

        logger.info("Total steps: %d, Warmup steps: %d", total_steps, warmup_steps)

        # 4. Training loop
        history: list[dict[str, Any]] = []
        self._best_f1 = 0.0

        for epoch in range(1, self.num_epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss = self._train_epoch(train_loader, optimizer, scheduler)

            # Validate
            val_metrics: dict[str, Any] = {}
            if epoch % self.eval_every_n_epochs == 0 or epoch == self.num_epochs:
                val_metrics = self._validate(val_loader)

                # Save best checkpoint
                if val_metrics.get("f1", 0.0) > self._best_f1:
                    self._best_f1 = val_metrics["f1"]
                    self._save_checkpoint(epoch)
                    logger.info("New best F1: %.4f — checkpoint saved", self._best_f1)

            epoch_duration = time.time() - epoch_start
            epoch_record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "duration_s": round(epoch_duration, 2),
                **{f"val_{k}": round(v, 6) if isinstance(v, float) else v for k, v in val_metrics.items()},
            }
            history.append(epoch_record)
            logger.info(
                "Epoch %d/%d — loss=%.4f  val_f1=%.4f  (%.1fs)",
                epoch, self.num_epochs, train_loss,
                val_metrics.get("f1", 0.0), epoch_duration,
            )

        # Save final model regardless
        self._save_checkpoint(self.num_epochs, tag="final")

        summary = {
            "best_val_f1": round(self._best_f1, 6),
            "total_epochs": self.num_epochs,
            "history": history,
            "save_dir": str(self.save_dir),
            "device": str(self.device),
        }
        logger.info("=== Training complete. Best F1: %.4f ===", self._best_f1)
        return summary

    def _train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
    ) -> float:
        """Run one training epoch. Returns average loss."""
        assert self._model is not None
        self._model.train()
        total_loss = 0.0
        num_steps = 0

        optimizer.zero_grad()
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self._model(**batch)
            loss = outputs.loss / self.gradient_accumulation_steps
            loss.backward()
            total_loss += outputs.loss.item()
            num_steps += 1

            if step % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # Handle remaining accumulated gradients
        if num_steps % self.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        return total_loss / max(num_steps, 1)

    @torch.no_grad()
    def _validate(self, val_loader: torch.utils.data.DataLoader) -> dict[str, Any]:
        """Run validation and compute seqeval metrics."""
        assert self._model is not None
        self._model.eval()

        all_preds: list[list[str]] = []
        all_labels: list[list[str]] = []
        total_loss = 0.0
        num_steps = 0

        for batch in val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self._model(**batch)
            total_loss += outputs.loss.item()
            num_steps += 1

            logits = outputs.logits  # (batch, seq_len, num_labels)
            predictions = torch.argmax(logits, dim=-1)  # (batch, seq_len)

            for pred_seq, label_seq in zip(predictions, batch["labels"]):
                pred_labels: list[str] = []
                true_labels: list[str] = []
                for pred_id, label_id in zip(pred_seq, label_seq):
                    if label_id.item() == -100:
                        continue
                    pred_labels.append(ID2LABEL.get(pred_id.item(), "O"))
                    true_labels.append(ID2LABEL.get(label_id.item(), "O"))
                if pred_labels:
                    all_preds.append(pred_labels)
                    all_labels.append(true_labels)

        avg_loss = total_loss / max(num_steps, 1)

        if all_preds and all_labels:
            f1 = seqeval_f1(all_labels, all_preds, average="micro", zero_division=0)
            try:
                report = classification_report(all_labels, all_preds, zero_division=0)
                logger.info("\n%s", report)
            except ValueError:
                logger.info("Val F1=%.4f (no entity spans found for detailed report)", f1)
        else:
            f1 = 0.0
            logger.warning("No valid predictions for seqeval — all tokens were ignored (-100).")

        return {"f1": f1, "loss": avg_loss, "num_samples": len(all_preds)}

    def _save_checkpoint(self, epoch: int, tag: str = "best") -> None:
        """Save model and processor to disk."""
        assert self._model is not None
        save_path = self.save_dir / tag
        save_path.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(save_path)
        logger.info("Checkpoint saved to %s (epoch %d)", save_path, epoch)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune LayoutLMv3 on CORD")
    parser.add_argument("--model-name", default="microsoft/layoutlmv3-base")
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--save-dir", default="experiments/artifacts/cord_finetuned")
    parser.add_argument("--eval-every-n-epochs", type=int, default=1)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer = LayoutLMv3Trainer(
        model_name=args.model_name,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_train_samples=args.max_train_samples,
        save_dir=args.save_dir,
        eval_every_n_epochs=args.eval_every_n_epochs,
        device=args.device,
    )
    summary = trainer.train()
    logger.info("Training summary: %s", summary)


if __name__ == "__main__":
    main()
