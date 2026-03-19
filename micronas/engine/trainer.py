import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from tqdm import tqdm
from utils.logger import get_logger

logger = get_logger("Trainer")

class Trainer:
    def __init__(self, model, train_loader, val_loader, task="classification"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.task = task

        # HuggingFace Accelerate for easy GPU/CPU offloading and mixed precision
        self.accelerator = Accelerator(mixed_precision="fp16" if torch.cuda.is_available() else "no")
        logger.info(f"Accelerator device: {self.accelerator.device}, mixed_precision: {self.accelerator.mixed_precision}")

        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.0005, weight_decay=1e-4)

        if self.task == "classification":
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()

        # Add Learning Rate Scheduler for Convergence
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2)

        # Prepare components for accelerate
        self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler
        )

        # Gradient Checkpointing Support (Memory Optimization)
        # Enable gradient checkpointing if model is Deep and Memory constrained
        # For simplicity, if CNN has many layers we can enable it, but
        # custom models need module wrappers for perfect checkpointing.
        # In this hackathon, mixed precision + accelerate mapping is standard.

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "best_val_loss": float('inf')
        }

    def train(self, epochs=5, callback=None, early_stopping_patience=3):
        logger.info(f"Starting training for {epochs} epochs on task: {self.task}")

        epochs_no_improve = 0
        best_loss = float('inf')

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0

            # Use fallback try-except for OOM catching per batch
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                try:
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)

                    if self.task == "regression":
                        targets = targets.view(-1, 1).float()

                    loss = self.criterion(outputs, targets)
                    self.accelerator.backward(loss)

                    # Gradient clipping to prevent exploding loss
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                    self.optimizer.step()

                    total_loss += loss.item()

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning(f"OOM caught in batch {batch_idx}! Attempting to clear cache and skip batch.")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        # Dynamic batch sizing or skipping
                        continue
                    else:
                        raise e

            avg_train_loss = total_loss / max(1, len(self.train_loader))
            self.history["train_loss"].append(avg_train_loss)

            # Validation
            val_loss, val_metric = self.evaluate()
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_metric)

            # Step the scheduler
            self.scheduler.step(val_loss)

            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Metric: {val_metric:.4f}")

            if callback:
                # Callback to GUI (epoch, train_loss, val_loss, val_metric)
                callback(epoch+1, avg_train_loss, val_loss, val_metric)

            # Early Stopping Check
            if val_loss < best_loss:
                best_loss = val_loss
                self.history["best_val_loss"] = best_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1} (No improvement for {early_stopping_patience} epochs).")
                    break

        self.history['preds'] = self.last_val_preds
        self.history['targets'] = self.last_val_targets
        logger.info("Training complete.")
        return self.history

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        correct_top1 = 0
        correct_top3 = 0
        total = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                outputs = self.model(inputs)

                if self.task == "regression":
                    targets = targets.view(-1, 1).float()
                    loss = self.criterion(outputs, targets)
                    # For regression, metric is negative MSE (or we just return MSE)
                    metric_val = -loss.item()
                else:
                    loss = self.criterion(outputs, targets)

                    # Top-1 Accuracy
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct_top1 += predicted.eq(targets).sum().item()

                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())

                    # Top-3 Accuracy
                    if outputs.size(1) >= 3:
                        _, top3_pred = outputs.topk(3, 1, True, True)
                        correct_top3 += top3_pred.eq(targets.view(-1, 1).expand_as(top3_pred)).sum().item()

                total_loss += loss.item()

        avg_loss = total_loss / max(1, len(self.val_loader))
        if self.task == "classification":
            metric = (correct_top1 / total) * 100 if total > 0 else 0
            if total > 0 and outputs.size(1) >= 3:
                top3_acc = (correct_top3 / total) * 100
                logger.info(f"Val Top-1 Acc: {metric:.2f}% | Val Top-3 Acc: {top3_acc:.2f}%")
        else:
            metric = avg_loss # MSE

        self.last_val_preds = all_preds
        self.last_val_targets = all_targets

        return avg_loss, metric
