"""Continuous-time 4D-CBCT helpers."""

from .dataset import ProjectionDataset
from .losses import (
    projection_mse_loss,
    reference_loss,
    signal_corr_loss,
    temporal_smoothness_loss,
    velocity_tv_smoothness_loss,
)
from .manifold import MotionManifoldAE, manifold_regularization_loss
from .model import ContinuousTimeField
from .renderer import project_volume_tigre_autograd, query_volume, query_volume_condition, render_ray_batch
from .shroud import amsterdam_shroud, predict_shroud_surrogate_from_model, predict_surrogate_from_model
from .svf import integrate_stationary_velocity

__all__ = [
    "ContinuousTimeField",
    "ProjectionDataset",
    "projection_mse_loss",
    "temporal_smoothness_loss",
    "reference_loss",
    "signal_corr_loss",
    "velocity_tv_smoothness_loss",
    "MotionManifoldAE",
    "manifold_regularization_loss",
    "integrate_stationary_velocity",
    "render_ray_batch",
    "query_volume",
    "query_volume_condition",
    "project_volume_tigre_autograd",
    "predict_surrogate_from_model",
    "predict_shroud_surrogate_from_model",
    "amsterdam_shroud",
]
