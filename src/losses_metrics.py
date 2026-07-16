from __future__ import annotations

"""Project tensor contract
-----------------------
- Model output: raw logits with shape [B, 1, H, W].
- Dataset target: non-negative saliency distribution with shape
  [B, 1, H, W].
- Every target map is already normalized in ``src.data`` so that
  its pixels sum to 1.
- Metrics receive probability distributions, not raw logits.
"""
from typing import Literal
import torch
import torch.nn.functional as F
from torch import Tensor

EPSILON = 1e-8
Reduction = Literal["mean", "none"]

# Check the structural contract of a batched saliency tensor.
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
    

# Redcuce one metric value per image to the requested output
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
# Softmax applied to every image indipendently
# logits -> raw model output, Log-Prob -> return with same shape of logits
def spatial_log_softmax(logits: Tensor,) -> Tensor:
    _check_map_tensor(logits, "logits")
    original_shape = logits.shape

    flat_logits = logits.flatten(start_dim=1).float()
    flat_log_prob = F.log_softmax(flat_logits, dim=1)
    return flat_log_prob.reshape(original_shape)


# Convert raw logirs into spatial prob distribution
def spatial_softmax(logits: Tensor, ) -> Tensor:
    return spatial_log_softmax(logits).exp()


"""Compute 
--- KLD(target || prediction) ---
Both tensor must already be valid spatial probability distribution.
Lower values are better.
---------------------------
prediction -> predicted prob maps 
target -> Ground-truth prob maps 
epsilon -> small value preventing log of 0
reduction -> mean / none
Returns -> Scalar tensor for "mean", tensor with shape [B] for "none"
----------------------------
"""
def kld_divergence(prediction: Tensor, target: Tensor, *, epsilon: float = EPSILON, reduction: Reduction = "mean", ) -> Tensor:
    
    _check_pair(prediction, target)
    prediction_flat = prediction.float().flatten(start_dim=1)
    target_flat = target.float().flatten(start_dim=1)

    per_image_kld = (target_flat * (torch.log(target_flat + epsilon) - torch.log(prediction_flat + epsilon))).sum(dim=1)

    return _reduce(per_image_kld, reduction,)


"""
--- Pearson Correlation Coeffienct ---
Computer per image, Higer value better.
Const map recives a score of zero becouse its spatial variance is zero
---------------------------
prediction -> predicted prob maps 
target -> Ground-truth prob maps 
epsilon -> small value preventing log of 0
reduction -> mean / none
Returns -> Scalar tensor for "mean", tensor with shape [B] for "none"
----------------------------
"""
def correlation_coefficent(prediction: Tensor, target: Tensor, *, epsilon: float = EPSILON, reduction: Reduction = "mean", ) -> Tensor:

    _check_pair(prediction, target)
    prediction_flat = prediction.float().flatten(start_dim=1)
    target_flat = target.float().flatten(start_dim=1)

    prediction_centered = (prediction_flat - prediction_flat.mean(dim = 1, keepdim=True,))
    target_centered = (target_flat - target_flat.mean(dim=1, keepdim=True))

    numerator = (prediction_centered * target_centered).sum(dim=1)
    denominator = torch.sqrt(prediction_centered.square().sum(dim=1) * target_centered.square().sum(dim=1)).clamp_min(epsilon)

    per_image_cc = numerator / denominator
    
    return _reduce(per_image_cc, reduction,)



"""
Similarity Score
Compute histogram-intersection similarity
Both tensor must already be valid spatial probability distributions
Higher values better -> perfect match SIM = 1
------------------------
predictions -> predicted prob maps 
target -> Ground-truth prob maps 
reduction -> mean / none
Return -> Scalar tensor for "mean", tensor with shape [B] for "none"
"""
def similarity_score(prediction: Tensor, target: Tensor, *, reduction: Reduction = "mean",) -> Tensor:
    _check_pair(prediction, target)
    prediction_flat = prediction.float().flatten(start_dim=1)
    target_flat = target.float().flatten(start_dim=1)

    per_image_similarity = torch.minimum(prediction_flat, target_flat).sum(dim=1)
    
    return _reduce(per_image_similarity, reduction,)
    

"""
Compute training objective for all neural models
---
loss = KLD(target||prediction) + cc_weight * (1 - CC(prediction, target))
---
target already normalized, raw model logits converted in spatial probability distribution
Log-Softmax is used directly for the KLD term to improve numerical stability
---------------------------
logits -> raw model output 
target -> Ground-truth prob maps 
cc_weight -> weight assigned to the corrrelation component
epsilon -> small value preventing log of 0
reduction -> mean / none
Returns -> Scalar tensor for "mean", tensor with shape [B] for "none"
----------------------------
"""
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

    per_image_cc = correlation_coefficent(prediction, target_float, epsilon=epsilon, reduction="none",)

    per_image_loss = (per_image_kld + cc_weight * (1.0 - per_image_cc))

    return _reduce(per_image_loss, reduction,)





