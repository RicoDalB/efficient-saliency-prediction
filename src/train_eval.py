from __future__ import annotations
# Training and evaluation helpers shared by all saliency models

import math
import os
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader

from src.losses_metrics import (
    correlation_coefficient,
    kld_divergence,
    saliency_loss,
    similarity_score,
    spatial_softmax,
    validate_spatial_distribution,
)


# Used only to undo ImageNet normalization in prediction panels.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# File and runtime helpers
def _prepare_output_file(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def get_git_commit(repo_root: str | Path | None) -> str | None:
    if repo_root is None:
        return None

    root = Path(repo_root)
    if not root.is_dir():
        return None

    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    commit = result.stdout.strip()
    return commit or None


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _atomic_torch_save(payload: Any, path: str | Path) -> None:
    final_path = _prepare_output_file(path)
    temporary_path = final_path.with_suffix(final_path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, final_path)


def _make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()

# Batch handling
def _move_batch_to_device(
    batch: Mapping[str, Any],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    if "image" not in batch or "target" not in batch:
        raise KeyError("Every batch must contain 'image' and 'target' entries.")

    images = batch["image"]
    targets = batch["target"]

    if not isinstance(images, Tensor) or not isinstance(targets, Tensor):
        raise TypeError("batch['image'] and batch['target'] must be tensors.")
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(
            "Images must have shape [B, 3, H, W], "
            f"but received {tuple(images.shape)}."
        )
    if targets.ndim != 4 or targets.shape[1] != 1:
        raise ValueError(
            "Targets must have shape [B, 1, H, W], "
            f"but received {tuple(targets.shape)}."
        )
    if images.shape[0] != targets.shape[0] or images.shape[-2:] != targets.shape[-2:]:
        raise ValueError(
            "Images and targets must have the same batch and spatial dimensions: "
            f"images={tuple(images.shape)}, targets={tuple(targets.shape)}."
        )

    images = images.to(device=device, non_blocking=True)
    targets = targets.to(device=device, non_blocking=True)
    return images, targets


def _extract_sample_ids(
    batch: Mapping[str, Any],
    batch_size: int,
    running_start_index: int,
) -> list[str]:
    raw_ids = batch.get("sample_id")

    if raw_ids is None:
        return [
            f"sample_{running_start_index + offset:06d}"
            for offset in range(batch_size)
        ]

    if isinstance(raw_ids, Tensor):
        values = raw_ids.detach().cpu().tolist()
        if not isinstance(values, list):
            values = [values]
    elif isinstance(raw_ids, (list, tuple)):
        values = list(raw_ids)
    else:
        values = [raw_ids]

    if len(values) != batch_size:
        raise ValueError(
            "Number of sample IDs does not match batch size: "
            f"ids={len(values)}, batch_size={batch_size}."
        )

    return [str(value) for value in values]


# Enable or disable gradient updates for the entire encoder.
def set_encoder_trainable(model: nn.Module, trainable: bool) -> None:
    if not hasattr(model, "encoder"):
        raise AttributeError("The model must expose an 'encoder' module.")

    for parameter in model.encoder.parameters():
        parameter.requires_grad = trainable


# Keep encoder BatchNorm running statistics fixed during training.
def freeze_encoder_batchnorm_statistics(model: nn.Module) -> None:
    if not hasattr(model, "encoder"):
        raise AttributeError("The model must expose an 'encoder' module.")

    for module in model.encoder.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


# Build the shared AdamW optimizer with two learning-rate groups.
def build_optimizer(
    model: nn.Module,
    *,
    encoder_lr: float = 1e-4,
    decoder_lr: float = 3e-4,
    weight_decay: float = 1e-4,
) -> AdamW:
    if not hasattr(model, "encoder") or not hasattr(model, "decoder"):
        raise AttributeError("The model must expose both 'encoder' and 'decoder'.")
    if encoder_lr <= 0 or decoder_lr <= 0:
        raise ValueError("encoder_lr and decoder_lr must be greater than zero.")
    if weight_decay < 0:
        raise ValueError("weight_decay cannot be negative.")

    encoder_parameters = list(model.encoder.parameters())
    decoder_parameters = list(model.decoder.parameters())

    if not encoder_parameters:
        raise ValueError("The encoder contains no parameters.")
    if not decoder_parameters:
        raise ValueError("The decoder contains no parameters.")

    return AdamW(
        [
            {
                "name": "encoder",
                "params": encoder_parameters,
                "lr": encoder_lr,
            },
            {
                "name": "decoder",
                "params": decoder_parameters,
                "lr": decoder_lr,
            },
        ],
        weight_decay=weight_decay,
    )


# Read the current encoder and decoder learning rates.
def get_learning_rates(optimizer: Optimizer) -> dict[str, float]:
    learning_rates: dict[str, float] = {}
    for index, group in enumerate(optimizer.param_groups):
        name = str(group.get("name", f"group_{index}"))
        learning_rates[name] = float(group["lr"])
    return learning_rates


# Train the model for exactly one complete epoch.
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    *,
    scaler: Any,
    use_amp: bool = True,
    gradient_clip_norm: float | None = 1.0,
    freeze_batchnorm: bool = True,
) -> dict[str, float]:
    if len(dataloader) == 0:
        raise ValueError("Training DataLoader is empty.")
    if gradient_clip_norm is not None and gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive or None.")

    model.train()
    if freeze_batchnorm:
        freeze_encoder_batchnorm_statistics(model)

    total_loss = 0.0
    total_samples = 0
    start_time = time.perf_counter()

    for batch in dataloader:
        images, targets = _move_batch_to_device(batch, device)
        batch_size = images.shape[0]

        optimizer.zero_grad(set_to_none=True)

        with _autocast_context(device, use_amp):
            logits = model(images)

            if logits.shape != targets.shape:
                raise ValueError(
                    "Model output and target shapes differ: "
                    f"logits={tuple(logits.shape)}, targets={tuple(targets.shape)}."
                )

            loss = saliency_loss(logits, targets)

        if not torch.isfinite(loss).item():
            raise FloatingPointError(
                "Training loss became NaN or infinite. "
                "Stop the run and inspect the current batch, targets, and learning rate."
            )

        scaler.scale(loss).backward()

        if gradient_clip_norm is not None:
            # Gradients must be unscaled before clipping, otherwise the clipping
            # threshold would be applied to artificially scaled AMP gradients.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ],
                max_norm=gradient_clip_norm,
            )

        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.detach()) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("Training epoch processed zero samples.")

    elapsed = time.perf_counter() - start_time
    return {
        "loss": total_loss / total_samples,
        "num_samples": float(total_samples),
        "num_batches": float(len(dataloader)),
        "duration_seconds": elapsed,
    }


# Evaluate a model on validation or test data without gradients.
@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool = True,
    return_per_image: bool = False,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if len(dataloader) == 0:
        raise ValueError("Evaluation DataLoader is empty.")

    model.eval()

    total_loss = 0.0
    total_kld = 0.0
    total_cc = 0.0
    total_sim = 0.0
    total_samples = 0
    per_image_records: list[dict[str, Any]] = []
    start_time = time.perf_counter()

    for batch in dataloader:
        images, targets = _move_batch_to_device(batch, device)
        batch_size = images.shape[0]

        with _autocast_context(device, use_amp):
            logits = model(images)

        if logits.shape != targets.shape:
            raise ValueError(
                "Model output and target shapes differ during evaluation: "
                f"logits={tuple(logits.shape)}, targets={tuple(targets.shape)}."
            )

        # Metric functions operate in float32 for numerical stability. 
        prediction = spatial_softmax(logits)
        validate_spatial_distribution(prediction, name="prediction")

        per_image_loss = saliency_loss(logits, targets, reduction="none")
        per_image_kld = kld_divergence(prediction, targets, reduction="none")
        per_image_cc = correlation_coefficient(prediction, targets, reduction="none")
        per_image_sim = similarity_score(prediction, targets, reduction="none")

        metric_tensors = {
            "loss": per_image_loss,
            "kld": per_image_kld,
            "cc": per_image_cc,
            "sim": per_image_sim,
        }
        for metric_name, values in metric_tensors.items():
            if not torch.isfinite(values).all().item():
                raise FloatingPointError(
                    f"Evaluation metric {metric_name!r} contains NaN or infinity."
                )

        total_loss += float(per_image_loss.sum())
        total_kld += float(per_image_kld.sum())
        total_cc += float(per_image_cc.sum())
        total_sim += float(per_image_sim.sum())

        if return_per_image:
            sample_ids = _extract_sample_ids(
                batch,
                batch_size=batch_size,
                running_start_index=total_samples,
            )
            for index, sample_id in enumerate(sample_ids):
                per_image_records.append(
                    {
                        "sample_id": sample_id,
                        "loss": float(per_image_loss[index].detach().cpu()),
                        "kld": float(per_image_kld[index].detach().cpu()),
                        "cc": float(per_image_cc[index].detach().cpu()),
                        "sim": float(per_image_sim[index].detach().cpu()),
                    }
                )

        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("Evaluation pass processed zero samples.")

    elapsed = time.perf_counter() - start_time
    summary = {
        "loss": total_loss / total_samples,
        "kld": total_kld / total_samples,
        "cc": total_cc / total_samples,
        "sim": total_sim / total_samples,
        "num_samples": float(total_samples),
        "num_batches": float(len(dataloader)),
        "duration_seconds": elapsed,
    }
    return summary, per_image_records


# Save a complete recoverable training checkpoint.
def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scaler: Any,
    epoch: int,
    best_val_cc: float,
    epochs_without_improvement: int,
    history: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any] | None = None,
    split_seed: int = 42,
    git_commit: str | None = None,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": int(epoch),
        "best_val_cc": float(best_val_cc),
        "epochs_without_improvement": int(epochs_without_improvement),
        "history": [_to_serializable(dict(row)) for row in history],
        "config": _to_serializable(dict(config or {})),
        "split_seed": int(split_seed),
        "git_commit": git_commit,
        "pytorch_version": torch.__version__,
    }
    _atomic_torch_save(checkpoint, path)


# Load a checkpoint into model, optimizer, and AMP scaler.
def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    device: torch.device,
    optimizer: Optimizer | None = None,
    scaler: Any | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Invalid checkpoint without model_state_dict: {checkpoint_path}")

    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, Tensor):
                    state[key] = value.to(device)

    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint


# Save the complete epoch history as a machine-readable CSV.
def save_history_csv(
    history: Sequence[Mapping[str, Any]],
    path: str | Path,
) -> Path:
    if not history:
        raise ValueError("Cannot save an empty training history.")

    output_path = _prepare_output_file(path)
    dataframe = pd.DataFrame([dict(row) for row in history])
    dataframe.to_csv(output_path, index=False)
    return output_path


# Plot all epoch lines needed to monitor convergence/overfitting.
def plot_training_history(
    history: Sequence[Mapping[str, Any]],
    *,
    output_path: str | Path | None = None,
    show: bool = True,
    title: str | None = None,
):
    if not history:
        raise ValueError("Cannot plot an empty training history.")

    dataframe = pd.DataFrame([dict(row) for row in history])
    required_columns = {
        "epoch",
        "train_loss",
        "val_loss",
        "val_kld",
        "val_cc",
        "val_sim",
    }
    missing = required_columns.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Training history is missing columns: {sorted(missing)}")

    epochs = dataframe["epoch"].to_numpy()

    figure, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True)

    axes[0].plot(epochs, dataframe["train_loss"], marker="o", label="Train loss")
    axes[0].plot(epochs, dataframe["val_loss"], marker="o", label="Validation loss")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training vs validation loss — overfitting monitor")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, dataframe["val_kld"], marker="o", label="Validation KLD")
    axes[1].set_ylabel("KLD (lower is better)")
    axes[1].set_title("Validation KLD")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(epochs, dataframe["val_cc"], marker="o", label="Validation CC")
    axes[2].plot(epochs, dataframe["val_sim"], marker="o", label="Validation SIM")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Score (higher is better)")
    axes[2].set_title("Validation quality")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    if title:
        figure.suptitle(title, fontsize=14)

    figure.tight_layout(rect=(0, 0, 1, 0.98 if title else 1))

    if output_path is not None:
        figure_path = _prepare_output_file(output_path)
        figure.savefig(figure_path, dpi=180, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(figure)

    return figure


# Save a fixed qualitative validation panel for the current model.
@torch.inference_mode()
def plot_prediction_panel(
    model: nn.Module,
    preview_batch: Mapping[str, Any],
    device: torch.device,
    *,
    output_path: str | Path | None = None,
    max_samples: int = 4,
    use_amp: bool = True,
    title: str | None = None,
):
    if max_samples <= 0:
        raise ValueError("max_samples must be greater than zero.")

    images, targets = _move_batch_to_device(preview_batch, device)
    number_of_samples = min(max_samples, images.shape[0])

    model.eval()
    with _autocast_context(device, use_amp):
        logits = model(images[:number_of_samples])
    predictions = spatial_softmax(logits).detach().cpu()

    images_cpu = images[:number_of_samples].detach().cpu()
    targets_cpu = targets[:number_of_samples].detach().cpu()

    mean = IMAGENET_MEAN.to(dtype=images_cpu.dtype)
    std = IMAGENET_STD.to(dtype=images_cpu.dtype)
    images_cpu = (images_cpu * std + mean).clamp(0.0, 1.0)

    figure, axes = plt.subplots(
        number_of_samples,
        3,
        figsize=(12, 3.6 * number_of_samples),
        squeeze=False,
    )

    for row in range(number_of_samples):
        axes[row, 0].imshow(images_cpu[row].permute(1, 2, 0))
        axes[row, 1].imshow(targets_cpu[row, 0], cmap="magma")
        axes[row, 2].imshow(predictions[row, 0], cmap="magma")

        axes[row, 0].set_title("RGB image")
        axes[row, 1].set_title("Ground truth")
        axes[row, 2].set_title("Prediction")

        for column in range(3):
            axes[row, column].axis("off")

    if title:
        figure.suptitle(title, fontsize=14)

    figure.tight_layout(rect=(0, 0, 1, 0.98 if title else 1))

    if output_path is not None:
        figure_path = _prepare_output_file(output_path)
        figure.savefig(figure_path, dpi=180, bbox_inches="tight")

    plt.show()
    plt.close(figure)
    return figure


# Detect a simple warning pattern consistent with overfitting.

def detect_possible_overfitting(
    history: Sequence[Mapping[str, Any]],
    *,
    window: int = 3,
) -> bool:
    if window < 2:
        raise ValueError("window must be at least 2.")
    if len(history) < window:
        return False

    recent = list(history)[-window:]
    train_losses = [float(row["train_loss"]) for row in recent]
    val_losses = [float(row["val_loss"]) for row in recent]
    val_cc = [float(row["val_cc"]) for row in recent]

    train_decreasing = all(
        later < earlier for earlier, later in zip(train_losses, train_losses[1:])
    )
    val_increasing = all(
        later > earlier for earlier, later in zip(val_losses, val_losses[1:])
    )
    cc_not_improving = max(val_cc[1:]) <= val_cc[0]

    return train_decreasing and val_increasing and cc_not_improving

# ---------------------------------------------------------------------------------------
# Run the complete multi-epoch training and validation procedure.
# ----------------------------------------------------------------------------------------
def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    *,
    model_name: str,
    epochs: int,
    checkpoint_dir: str | Path,
    history_csv_path: str | Path,
    curves_figure_path: str | Path,
    preview_batch: Mapping[str, Any] | None = None,
    preview_figure_path: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
    split_seed: int = 42,
    repo_root: str | Path | None = None,
    patience: int = 3,
    min_delta: float = 1e-4,
    freeze_encoder_epochs: int = 1,
    freeze_encoder_batchnorm: bool = True,
    use_amp: bool = True,
    gradient_clip_norm: float | None = 1.0,
    resume_from: str | Path | None = None,
    show_curves_each_epoch: bool = True,
) -> list[dict[str, Any]]:
    if epochs <= 0:
        raise ValueError("epochs must be greater than zero.")
    if patience <= 0:
        raise ValueError("patience must be greater than zero.")
    if min_delta < 0:
        raise ValueError("min_delta cannot be negative.")
    if freeze_encoder_epochs < 0:
        raise ValueError("freeze_encoder_epochs cannot be negative.")

    checkpoint_root = Path(checkpoint_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_root / "best.pt"
    last_checkpoint_path = checkpoint_root / "last.pt"

    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = _make_grad_scaler(enabled=amp_enabled)
    git_commit = get_git_commit(repo_root)

    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_val_cc = -math.inf
    epochs_without_improvement = 0

    if resume_from is not None:
        checkpoint = load_checkpoint(
            resume_from,
            model=model,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
        )
        completed_epoch = int(checkpoint.get("epoch", 0))
        start_epoch = completed_epoch + 1
        best_val_cc = float(checkpoint.get("best_val_cc", -math.inf))
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        history = [dict(row) for row in checkpoint.get("history", [])]

        print(
            f"Resumed {model_name} from epoch {completed_epoch}. "
            f"Next epoch: {start_epoch}; best validation CC: {best_val_cc:.4f}."
        )

    if start_epoch > epochs:
        print(
            f"Checkpoint already completed epoch {start_epoch - 1}, which is "
            f"not smaller than requested maximum epochs={epochs}."
        )
        return history

    run_start_time = time.perf_counter()

    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.perf_counter()

        encoder_is_trainable = epoch > freeze_encoder_epochs
        set_encoder_trainable(model, trainable=encoder_is_trainable)

        encoder_status = "trainable" if encoder_is_trainable else "frozen"
        print("\n" + "=" * 78)
        print(
            f"{model_name} | epoch {epoch}/{epochs} | encoder: {encoder_status} | "
            f"AMP: {amp_enabled}"
        )
        print("=" * 78)

        train_summary = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler=scaler,
            use_amp=amp_enabled,
            gradient_clip_norm=gradient_clip_norm,
            freeze_batchnorm=freeze_encoder_batchnorm,
        )

        val_summary, _ = evaluate_model(
            model,
            val_loader,
            device,
            use_amp=amp_enabled,
            return_per_image=False,
        )

        learning_rates = get_learning_rates(optimizer)
        epoch_duration = time.perf_counter() - epoch_start_time

        history_row: dict[str, Any] = {
            "model": model_name,
            "epoch": epoch,
            "train_loss": train_summary["loss"],
            "val_loss": val_summary["loss"],
            "val_kld": val_summary["kld"],
            "val_cc": val_summary["cc"],
            "val_sim": val_summary["sim"],
            "encoder_lr": learning_rates.get("encoder", float("nan")),
            "decoder_lr": learning_rates.get("decoder", float("nan")),
            "encoder_trainable": encoder_is_trainable,
            "train_seconds": train_summary["duration_seconds"],
            "val_seconds": val_summary["duration_seconds"],
            "epoch_time_seconds": epoch_duration,
        }
        history.append(history_row)

        current_val_cc = float(val_summary["cc"])
        improved = current_val_cc > best_val_cc + min_delta

        if improved:
            best_val_cc = current_val_cc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Persist numerical history immediately before checkpointing and plotting.
        save_history_csv(history, history_csv_path)

        save_checkpoint(
            last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            best_val_cc=best_val_cc,
            epochs_without_improvement=epochs_without_improvement,
            history=history,
            config=config,
            split_seed=split_seed,
            git_commit=git_commit,
        )

        if improved:
            save_checkpoint(
                best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                best_val_cc=best_val_cc,
                epochs_without_improvement=epochs_without_improvement,
                history=history,
                config=config,
                split_seed=split_seed,
                git_commit=git_commit,
            )

        print(
            f"Train loss: {train_summary['loss']:.4f} | "
            f"Val loss: {val_summary['loss']:.4f} | "
            f"Val KLD: {val_summary['kld']:.4f} | "
            f"Val CC: {val_summary['cc']:.4f} | "
            f"Val SIM: {val_summary['sim']:.4f}"
        )
        print(
            f"Epoch time: {epoch_duration / 60.0:.2f} min | "
            f"Best val CC: {best_val_cc:.4f} | "
            f"No-improvement counter: {epochs_without_improvement}/{patience}"
        )
        if improved:
            print(f"New best checkpoint saved to: {best_checkpoint_path}")
        print(f"Last checkpoint saved to: {last_checkpoint_path}")
        print(f"History CSV updated at: {Path(history_csv_path)}")

        plot_training_history(
            history,
            output_path=curves_figure_path,
            show=show_curves_each_epoch,
            title=f"{model_name} training history",
        )

        if preview_batch is not None:
            plot_prediction_panel(
                model,
                preview_batch,
                device,
                output_path=preview_figure_path,
                max_samples=4,
                use_amp=amp_enabled,
                title=f"{model_name} validation predictions — epoch {epoch}",
            )

        if len(history) >= 3 and detect_possible_overfitting(history, window=3):
            print(
                "WARNING: recent curves show decreasing training loss together "
                "with increasing validation loss and no CC improvement. "
                "This pattern is consistent with possible overfitting."
            )

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping: validation CC did not improve by at least "
                f"{min_delta:g} for {patience} consecutive epochs."
            )
            break

    total_minutes = (time.perf_counter() - run_start_time) / 60.0
    print("\n" + "=" * 78)
    print(
        f"Training finished for {model_name}. "
        f"Best validation CC: {best_val_cc:.4f}. "
        f"Elapsed time: {total_minutes:.2f} minutes."
    )
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Last checkpoint: {last_checkpoint_path}")
    print("=" * 78)

    return history


# Save final dataset-level and per-image evaluation results.
def evaluate_and_save(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    model_name: str,
    summary_csv_path: str | Path,
    per_image_csv_path: str | Path | None = None,
    use_amp: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    summary, records = evaluate_model(
        model,
        dataloader,
        device,
        use_amp=use_amp,
        return_per_image=True,
    )

    summary_row: dict[str, Any] = {
        "model": model_name,
        "loss": summary["loss"],
        "kld": summary["kld"],
        "cc": summary["cc"],
        "sim": summary["sim"],
        "num_samples": int(summary["num_samples"]),
        "duration_seconds": summary["duration_seconds"],
    }

    summary_path = _prepare_output_file(summary_csv_path)
    pd.DataFrame([summary_row]).to_csv(summary_path, index=False)

    per_image_dataframe = pd.DataFrame(records)
    if not per_image_dataframe.empty:
        per_image_dataframe.insert(0, "model", model_name)

    if per_image_csv_path is not None:
        per_image_path = _prepare_output_file(per_image_csv_path)
        per_image_dataframe.to_csv(per_image_path, index=False)

    return summary_row, per_image_dataframe