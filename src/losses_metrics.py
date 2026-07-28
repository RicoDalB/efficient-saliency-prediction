from __future__ import annotations
# Losses, metrics, and spatial-distribution utilities for saliency prediction

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

EPSILON = 1e-8
Reduction = Literal["mean", "none"]

# Check the expected [B, 1, H, W] saliency-map shape
def _check_map_tensor(maps: Tensor, name: str ) -> None:
    if maps.ndim != 4:
        raise ValueError(f"{name} must have shape [B, 1, H, W], " f"but received {tuple(maps.shape)}.")

    if maps.shape[1] != 1:
        raise ValueError(f"{name} must contain exactly one saliency channel, " f"but received shape {tuple(maps.shape)}.")
    

# Check prediction and target can be compared 
def _check_pair(prediction: Tensor, target: Tensor,) -> None:
    _check_map_tensor(prediction, "prediction")
    _check_map_tensor(target, "target")
    if prediction.shape != target.shape:
        raise ValueError("Prediciton and target must have same shape:" f"Prediction: {tuple(prediction.shape)}" f"Target: {tuple(target.shape)}.")
    

# Apply metric reduction
def _reduce(values: Tensor, reduction: Reduction,) -> Tensor:
    if reduction == "none": return values
    if reduction == "mean": return values.mean()
    raise ValueError(f"Unsupported reduction {reduction!r}")


# Verify every sailency map is valid spatial distriburion
def validate_spatial_distribution(maps: Tensor, *, name: str = "maps", atol: float = 1e-5,) -> None:
    _check_map_tensor(maps, name)
    if not torch.isfinite(maps).all().item():
        raise ValueError(f"{name} contains NaN or infinite values")
    if (maps < 0).any().item():
        raise ValueError(f"{name} contains negative values")
    
    masses = maps.flatten(start_dim=1).sum(dim=1)
    excepted = torch.ones_like(masses)

    if not torch.allclose(masses, excepted, atol=atol, rtol=0.0):
        raise ValueError("Every map must sum to 1"
                         f"{masses.detach().cpu().tolist()}")
    

# Convert Raw logits into log-probabilities over image location
def spatial_log_softmax(logits: Tensor,) -> Tensor:
    _check_map_tensor(logits, "logits")
    original_shape = logits.shape

    flat_logits = logits.flatten(start_dim=1).float()
    flat_log_prob = F.log_softmax(flat_logits, dim=1)
    return flat_log_prob.reshape(original_shape)


# Convert raw logirs into spatial prob distribution
def spatial_softmax(logits: Tensor, ) -> Tensor:
    return spatial_log_softmax(logits).exp()


# Compute KLD(target || prediction). Lower values better
def kld_divergence(prediction: Tensor, target: Tensor, *, epsilon: float = EPSILON, reduction: Reduction = "mean", ) -> Tensor:
    
    _check_pair(prediction, target)
    prediction_flat = prediction.float().flatten(start_dim=1)
    target_flat = target.float().flatten(start_dim=1)

    per_image_kld = (target_flat * (torch.log(target_flat + epsilon) - torch.log(prediction_flat + epsilon))).sum(dim=1)

    return _reduce(per_image_kld, reduction,)


# Compute the Pearson Correlation Coeffienct for each image
def correlation_coefficient(prediction: Tensor, target: Tensor, *, epsilon: float = EPSILON, reduction: Reduction = "mean", ) -> Tensor:

    _check_pair(prediction, target)
    prediction_flat = prediction.float().flatten(start_dim=1)
    target_flat = target.float().flatten(start_dim=1)

    prediction_centered = (prediction_flat - prediction_flat.mean(dim = 1, keepdim=True,))
    target_centered = (target_flat - target_flat.mean(dim=1, keepdim=True))

    numerator = (prediction_centered * target_centered).sum(dim=1)
    denominator = torch.sqrt(prediction_centered.square().sum(dim=1) * target_centered.square().sum(dim=1)).clamp_min(epsilon)

    per_image_cc = numerator / denominator
    
    return _reduce(per_image_cc, reduction,)


# Compute histogram-intersection similarit
def similarity_score(prediction: Tensor, target: Tensor, *, reduction: Reduction = "mean",) -> Tensor:
    _check_pair(prediction, target)
    prediction_flat = prediction.float().flatten(start_dim=1)
    target_flat = target.float().flatten(start_dim=1)

    per_image_similarity = torch.minimum(prediction_flat, target_flat).sum(dim=1)
    
    return _reduce(per_image_similarity, reduction,)
    

# Compute the shared KLD and correlation training objective
def saliency_loss(logits: Tensor, target: Tensor, *, cc_weight: float = 0.5, epsilon: float = EPSILON, reduction: Reduction = "mean",) -> Tensor:

    _check_pair(logits, target)
    if cc_weight < 0: raise ValueError("cc_weights must be non negative")

    log_prediction = spatial_log_softmax(logits)
    prediction = log_prediction.exp()
    target_float = target.float()

    target_flat = target_float.flatten(start_dim=1)
    log_prediction_flat = log_prediction.flatten(start_dim=1)

    # Stable computation of KLD 
    per_image_kld = (target_flat * (torch.log(target_flat + epsilon) - log_prediction_flat)).sum(dim=1)

    per_image_cc = correlation_coefficient(prediction, target_float, epsilon=epsilon, reduction="none",)

    per_image_loss = (per_image_kld + cc_weight * (1.0 - per_image_cc))

    return _reduce(per_image_loss, reduction,)





