# RL2gMCP/agent.py
# Author: Xin-Wei Huang
# Updated: 2025-07-16

"""
This file defines the Agent class, which represents the core policy network
of the reinforcement learning system.

Classes:
- Agent: A PyTorch nn.Module that learns a policy to generate graphical procedures.
         Its forward pass is now simplified, delegating complex processing logic
         to the action_processor module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
import numpy as np

# Import the new action processor module
from . import action_processor

class Agent(nn.Module):
    """
    The Reinforcement Learning Agent, implemented as a configurable neural network.
    It is enhanced with Dropout for regularization and delegates action processing
    to a separate module for cleaner code.

    Attributes:
        m (int): The number of hypotheses.
        use_layer_norm (bool): Flag to use Layer Normalization.
        top_k (int | None): Sparsity parameter for the T matrix.
        dropout_rate (float | None): Dropout probability for regularization.
        base_layer (nn.Sequential): The shared hidden layers of the network.
        alpha_head (nn.Linear): The output layer for alpha_weight logits.
        T_head (nn.Linear): The output layer for T matrix logits.
    """
    def __init__(
            self,
            m: int,
            hidden_layers: list = [128, 64],
            activation_fn=nn.ReLU,
            use_layer_norm: bool = True,
            top_k: int = None,
            dropout_rate: float | None = None
    ):
        """
        Initializes the Agent.

        Args:
            m (int): The number of hypotheses.
            hidden_layers (list, optional): Sizes of hidden layers. Defaults to [128, 64].
            activation_fn (nn.Module, optional): Activation function. Defaults to nn.ReLU.
            use_layer_norm (bool, optional): Whether to use LayerNorm. Defaults to True.
            top_k (int, optional): Sparsity parameter. Defaults to None.
            dropout_rate (float | None, optional): Dropout probability. Defaults to None.
        """
        super(Agent, self).__init__()
        self.m = m
        self.use_layer_norm = use_layer_norm
        self.top_k = top_k
        self.dropout_rate = dropout_rate

        # --- Section 1: Programmatically build the base network layers ---
        layers = []
        in_features = 1
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))
            if self.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(activation_fn())
            # Add Dropout layer after activation for regularization
            if self.dropout_rate is not None and self.dropout_rate > 0:
                layers.append(nn.Dropout(p=self.dropout_rate))
            in_features = hidden_size
        self.base_layer = nn.Sequential(*layers)

        # --- Section 2: Define specialized output heads ---
        last_layer_size = hidden_layers[-1] if hidden_layers else 1
        self.alpha_head = nn.Linear(last_layer_size, m)
        self.T_head = nn.Linear(last_layer_size, m * m)

    def forward(
        self,
        deterministic: bool = False,
        alpha_weight_fix: torch.Tensor | None = None,
        T_fix: torch.Tensor | None = None
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Generates an action by producing logits and delegating processing.

        Args:
            deterministic (bool, optional): If True, take the mean of distributions. Defaults to False.
            alpha_weight_fix (torch.Tensor | None, optional): Constraints for alpha_weight. Defaults to None.
            T_fix (torch.Tensor | None, optional): Constraints for T matrix. Defaults to None.

        Returns:
            tuple[torch.Tensor, ...]: A tuple containing:
                - alpha_weight (torch.Tensor): The final weight vector.
                - T_sample (torch.Tensor): The final transition matrix.
                - log_prob (torch.Tensor): The total log probability of the action.
                - entropy (torch.Tensor): The total entropy of the policy.
        """
        # --- Section 1: Get raw logits from the network heads ---
        device = next(self.parameters()).device
        dummy_input = torch.ones(1, 1, device=device)
        x = self.base_layer(dummy_input)
        alpha_logits = self.alpha_head(x).squeeze(0)
        T_logits = self.T_head(x).view(self.m, self.m)

        # --- Section 2: Delegate processing to the action_processor module ---
        alpha_weight, log_prob_alpha, entropy_alpha = action_processor.process_alpha_weight(
            alpha_logits, alpha_weight_fix, deterministic
        )
        T_sample_raw, log_prob_T, entropy_T = action_processor.process_T_matrix(
            T_logits, T_fix, self.top_k, deterministic
        )

        # --- Section 3: Combine results for the final loss calculation ---
        log_prob = log_prob_alpha + log_prob_T
        entropy = entropy_alpha + entropy_T

        # --- Section 4: Final post-processing for T matrix (safety checks) ---
        T_sample = T_sample_raw.clone()
        T_sample.fill_diagonal_(0.0) # Ensure diagonal is exactly zero
        row_sums = T_sample.sum(dim=1, keepdim=True)
        row_sums[row_sums < 1e-9] = 1.0 # Avoid division by zero
        T_sample = T_sample / row_sums

        return alpha_weight, T_sample, log_prob, entropy
