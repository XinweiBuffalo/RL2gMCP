# RL2gMCP/utils.py
# Author: Xin-Wei Huang
# Updated: 2025-07-24

"""
This file contains the Hyperparameters class, which serves as a centralized
configuration hub for the entire project. All experimental settings, model
architectures, and training parameters are defined here to promote clarity,
reproducibility, and ease of modification.
"""

import torch
import torch.nn as nn
import numpy as np


class Hyperparameters:
    """
    A class to hold all hyperparameters for an experiment.
    This allows for easy access and modification of settings from a single source.
    """
    # ==========================================================================
    # I. Experiment Scenario Settings
    # ==========================================================================
    MARGINAL_POWER: list = [0.95, 0.9, 0.85, 0.65, 0.6]
    POWER_RANGE: float = 0.05
    REWARD_WEIGHTS: list = [20.0, 6.0, 2.0, 1.0, 1.0]

    # ==========================================================================
    # II. Reward Function Settings
    # ==========================================================================
    REWARD_TYPE: str = "psr"  # "avg_power" or "psr"

    # Settings for the "penalty_based" reward type
    PRIMARY_ENDPOINTS: list = [1, 0, 0, 0, 0]
    PRIMARY_RULE: str = "PRIMARY"
    PSR_THRESHOLD: float = 0.90
    PSR_PENALTY: float = 10.0
    W_PSR: float = 10.0
    W_SAP: float = 1.0

    # Settings for the "lexicographic" reward type (Approach 5)
    LEXI_M: float = 100.0

    # ==========================================================================
    # III. P-value Simulation Settings
    # ==========================================================================
    CORR_TYPE: str | None = "AR1"
    CORR_RHO: float | None = 0.6

    # ==========================================================================
    # IV. Model & Training Settings
    # ==========================================================================
    ALPHA_0: float = 0.05
    LEARNING_RATE: float = 0.0001
    EPISODES: int = 4000
    BATCH_SIZE: int = 128
    BASELINE_DECAY: float = 0.99
    DTYPE = torch.float32
    ENTROPY_COEFF: float = 0.001
    NUM_SIMULATIONS: int = 10000

    RANDOM_SEED: int = 10001

    DEVICE: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================================================
    # V. Agent Architecture Settings
    # ==========================================================================
    AGENT_HIDDEN_LAYERS: list = [128, 64]
    AGENT_ACTIVATION_FN = nn.ReLU
    AGENT_USE_LAYER_NORM: bool = True
    AGENT_TOP_K: int | None = None
    AGENT_DROPOUT_RATE: float | None = 0.2

    # ==========================================================================
    # VI. Optimizer Settings
    # ==========================================================================
    OPTIMIZER_TYPE: str = "AdamW"
    WEIGHT_DECAY: float = 1e-2

    # ==========================================================================
    # VII. Constrained Optimization Settings (optional)
    # ==========================================================================
    ALPHA_WEIGHT_FIX: list | None = None
    T_FIX: list[list] | None = None

