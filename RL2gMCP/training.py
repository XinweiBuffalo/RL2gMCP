# RL2gMCP/training.py
# Author: Xin-Wei Huang
# Updated: 2025-07-16

"""
This file contains the main training loop and associated helper functions
for the reinforcement learning agent.

Functions:
- plot_training_dashboard: Creates a 3x2 dashboard of key training metrics.
- calculate_...: Helper functions to compute performance metrics like PSR and SAP.
- calculate_learning_reward: A modular function to calculate the learning signal
                             based on the chosen reward type.
- train_agent: The main function that orchestrates the entire training process.
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.distributions import kl_divergence
import torch.nn.functional as F

from .gMCP import gMCP
from .agent import Agent
from .utils import Hyperparameters


# --- Helper Functions for Metric Calculation ---
def calculate_primary_success_rate(decisions_batch, primary_mask, primary_rule, device):
    # ... (code remains the same)
    primary_mask = primary_mask.to(device)
    num_primary = torch.sum(primary_mask)
    if num_primary == 0: return 1.0
    num_rejected = torch.sum(decisions_batch * primary_mask, dim = 1)
    if primary_rule == "CO_PRIMARY":
        successes = (num_rejected == num_primary).float()
    elif primary_rule == "DUAL_PRIMARY":
        successes = (num_rejected >= 1).float()
    elif primary_rule == "PRIMARY":
        if num_primary != 1: raise ValueError(
            f"PRIMARY_RULE is 'PRIMARY', but {num_primary.item()} primary endpoints found.")
        successes = (num_rejected >= 1).float()
    else:
        raise ValueError(f"Unsupported PRIMARY_RULE: '{primary_rule}'.")
    return successes.mean().item()


def calculate_secondary_power(decisions_batch, primary_mask, weights, device):
    # ... (code remains the same)
    primary_mask = primary_mask.to(device)
    weights = weights.to(device)
    secondary_mask = 1.0 - primary_mask
    secondary_weights = weights * secondary_mask
    sum_secondary_weights = torch.sum(secondary_weights)
    if sum_secondary_weights <= 1e-9: return 0.0
    normalized_secondary_weights = secondary_weights / sum_secondary_weights
    secondary_power_batch = decisions_batch @ normalized_secondary_weights
    return secondary_power_batch.mean().item()


# --- Centralized Reward Calculation Function ---
def calculate_learning_reward(config, decisions_batch, W_scaled, primary_mask, W, device):
    """
    Calculates the learning reward and all diagnostic metrics based on the REWARD_TYPE.

    Args:
        config (Hyperparameters): The experiment configuration object.
        decisions_batch (torch.Tensor): A batch of decision vectors.
        W_scaled (torch.Tensor): The scaled reward weights for Avg Power calculation.
        primary_mask (torch.Tensor): The binary mask for primary endpoints.
        W (torch.Tensor): The original reward weights for SAP calculation.
        device (torch.device): The computation device.

    Returns:
        tuple[float, dict]: A tuple containing:
            - learning_reward (float): The final scalar reward signal for the agent.
            - metrics (dict): A dictionary of all calculated diagnostic metrics.
    """
    metrics = {}

    # Always calculate all metrics for the dashboard, regardless of reward type
    metrics['psr'] = calculate_primary_success_rate(decisions_batch, primary_mask, config.PRIMARY_RULE, device)
    metrics['sap'] = calculate_secondary_power(decisions_batch, primary_mask, W, device)
    metrics['avg_reward'] = (decisions_batch.float() @ W_scaled.to(device)).mean().item()

    # --- Reward Switchboard: Selects the learning signal ---
    if config.REWARD_TYPE == "psr":
        penalty = config.PSR_PENALTY if metrics['psr'] < config.PSR_THRESHOLD else 0.0
        learning_reward = (config.W_PSR * metrics['psr']) + (config.W_SAP * metrics['sap']) - penalty
    elif config.REWARD_TYPE == "lexicographic":
        # Approach 5: R = M * min(PSR, t_P) + SAP
        capped_psr = min(metrics['psr'], config.PSR_THRESHOLD)
        learning_reward = config.LEXI_M * capped_psr + metrics['sap']
    elif config.REWARD_TYPE == "avg_power":
        learning_reward = metrics['avg_reward']
    else:
        raise ValueError(f"Unsupported REWARD_TYPE: '{config.REWARD_TYPE}'.")

    return learning_reward, metrics


# --- Plotting Function ---
def plot_training_dashboard(history: dict, output_path: str | None):

    fig, ax = plt.subplots(3, 2, figsize = (16, 20))
    fig.suptitle('Training Diagnostic Super-Dashboard', fontsize = 20, fontweight = 'bold')

    def plot_metric(ax_obj, data, title, ylabel, y_lim = None, log_scale = False):
        ax_obj.plot(data, label = 'Episode Value', alpha = 0.6, color = 'c')
        if len(data) >= 100:
            import pandas as pd
            smoothed = pd.Series(data).rolling(window = 100, min_periods = 1).mean()
            ax_obj.plot(smoothed, label = 'Smoothed (100 ep.)', color = 'b')
        ax_obj.set_title(title, fontsize = 14)
        ax_obj.set_xlabel('Episode', fontsize = 12)
        ax_obj.set_ylabel(ylabel, fontsize = 12)
        if y_lim: ax_obj.set_ylim(y_lim)
        if log_scale:
            ax_obj.set_yscale('log')
            ax_obj.set_ylabel(f"{ylabel} (log scale)", fontsize = 12)
        ax_obj.legend()
        ax_obj.grid(True, which = "both", ls = "-", alpha = 0.5)

    plot_metric(ax[0, 0], history['avg_reward'], 'PEP Reward', 'Avg. Scaled Power')
    plot_metric(ax[0, 1], history['psr'], 'Primary Power ($\\Pi_\\mathcal{P}$)', 'Success Rate', y_lim = [0, 1.05])
    plot_metric(ax[1, 0], history['sap'], 'Secondary Average Power (SAP)', 'Avg. Power', y_lim = [0, 1.05])
    plot_metric(ax[1, 1], history['entropy'], 'Policy Entropy', 'Entropy')
    if len(history['kl_divergence']) > 1:
        plot_metric(ax[2, 0], history['kl_divergence'][1:], 'Policy KL Divergence', 'KL Divergence', log_scale = True)
    plot_metric(ax[2, 1], history['loss'], 'Loss Function', 'Loss')
    plt.tight_layout(rect = [0, 0.03, 1, 0.96])
    if output_path:
        try:
            plt.savefig(f"{output_path}_training_dashboard.png", dpi = 300)
            print(f"\nSuccessfully saved training dashboard.")
        except Exception as e:
            print(f"\nError saving training dashboard: {e}")
    plt.show()


def train_agent(
        config: Hyperparameters,
        progress_gap: int = 200,
        output_path: str = None
):
    """
    The main training loop orchestrating the RL process.
    """
    # --- Section 1: Initialization ---
    start_time = time.time()
    device = config.DEVICE

    if config.RANDOM_SEED is not None:
        print(f"Using fixed random seed: {config.RANDOM_SEED}")
        np.random.seed(config.RANDOM_SEED)
        torch.manual_seed(config.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.RANDOM_SEED)

    m = len(config.MARGINAL_POWER)

    # This function call's inputs are defined in the config object.
    # Example: m=5, hidden_layers=[128, 64], dropout_rate=0.2
    # It returns an initialized Agent object.
    agent = Agent(m = m, hidden_layers = config.AGENT_HIDDEN_LAYERS, activation_fn = config.AGENT_ACTIVATION_FN,
                  use_layer_norm = config.AGENT_USE_LAYER_NORM, top_k = config.AGENT_TOP_K,
                  dropout_rate = config.AGENT_DROPOUT_RATE).to(device)

    optimizer_name = config.OPTIMIZER_TYPE.lower()
    if optimizer_name == 'adamw':
        optimizer = optim.AdamW(agent.parameters(), lr = config.LEARNING_RATE, weight_decay = config.WEIGHT_DECAY)
    else:
        optimizer = optim.Adam(agent.parameters(), lr = config.LEARNING_RATE)

    W = torch.tensor(config.REWARD_WEIGHTS, dtype = config.DTYPE, device = device)
    W_sum = torch.sum(W)
    W_scaled = W / W_sum if W_sum > 0 else W
    primary_mask = torch.tensor(config.PRIMARY_ENDPOINTS, dtype = config.DTYPE, device = device)
    cov_matrix = gMCP.create_covariance_matrix(m = m, corr_type = config.CORR_TYPE, rho = config.CORR_RHO)
    alpha_weight_fix_tensor = torch.tensor(config.ALPHA_WEIGHT_FIX, dtype = config.DTYPE,
                                           device = device) if config.ALPHA_WEIGHT_FIX is not None else None
    T_fix_tensor = torch.tensor(config.T_FIX, dtype = config.DTYPE,
                                device = device) if config.T_FIX is not None else None

    history = {'avg_reward': [], 'psr': [], 'sap': [], 'entropy': [], 'kl_divergence': [], 'loss': []}
    reward_baseline = 0.0
    old_alpha_dist, old_T_dists = None, None

    print(f"Starting training with REWARD_TYPE = '{config.REWARD_TYPE}'...")

    # --- Section 2: Main Training Loop ---
    for episode in range(config.EPISODES):
        agent.train()

        # --- Subsection 2.1: KL Divergence Calculation ---
        with torch.no_grad():
            x = agent.base_layer(torch.ones(1, 1, device = device))
            alpha_logits, T_logits = agent.alpha_head(x).squeeze(0), agent.T_head(x).view(m, m)
        current_alpha_dist = torch.distributions.Dirichlet(F.softplus(alpha_logits) + 1e-6)
        current_T_dists = [torch.distributions.Dirichlet(F.softplus(T_logits[i]) + 1e-6) for i in range(m)]
        kl_div = 0.0
        if old_alpha_dist:
            kl_alpha = kl_divergence(current_alpha_dist, old_alpha_dist).item()
            kl_T = sum(kl_divergence(current_T_dists[i], old_T_dists[i]).item() for i in range(m))
            kl_div = kl_alpha + kl_T
        history['kl_divergence'].append(kl_div)
        old_alpha_dist, old_T_dists = current_alpha_dist, current_T_dists

        # --- Subsection 2.2: Action Generation and Evaluation ---
        # This function call's inputs are defined in the config object.
        # It returns the generated graph (alpha_weight, T) and policy info (log_prob, entropy).
        alpha_weight, T, log_prob, entropy = agent(alpha_weight_fix = alpha_weight_fix_tensor, T_fix = T_fix_tensor)

        p_value_batch = gMCP.generate_p_values(batch_size = config.BATCH_SIZE, marginal_power = config.MARGINAL_POWER,
                                               power_range = config.POWER_RANGE, cov_matrix = cov_matrix,
                                               sim_alpha = config.ALPHA_0).to(dtype = config.DTYPE, device = device)
        decisions_batch = torch.stack(
            [gMCP.graphTest(alpha_weight.detach(), T.detach(), p, config.ALPHA_0) for p in p_value_batch])

        # --- Subsection 2.3: Reward Calculation ---
        # This function call calculates the learning signal and all diagnostic metrics.
        # It returns the final scalar reward and a dictionary of metrics.
        learning_reward, metrics = calculate_learning_reward(
            config, decisions_batch, W_scaled, primary_mask, W, device
        )

        # --- Subsection 2.4: Agent Update ---
        if episode == 0:
            reward_baseline = learning_reward
        else:
            reward_baseline = config.BASELINE_DECAY * reward_baseline + (1 - config.BASELINE_DECAY) * learning_reward
        advantage = learning_reward - reward_baseline

        # Entropy coefficient: support annealing via ENTROPY_COEFF_FINAL
        entropy_coeff_now = config.ENTROPY_COEFF
        if hasattr(config, 'ENTROPY_COEFF_FINAL') and config.ENTROPY_COEFF_FINAL is not None:
            frac = episode / max(config.EPISODES - 1, 1)
            entropy_coeff_now = config.ENTROPY_COEFF * (1 - frac) + config.ENTROPY_COEFF_FINAL * frac

        loss = -log_prob * advantage - entropy_coeff_now * entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # --- Subsection 2.5: Record History ---
        for key, value in metrics.items():
            history[key].append(value)
        history['entropy'].append(entropy.item())
        history['loss'].append(loss.item())

        if (episode + 1) % progress_gap == 0:
            print(
                f"Ep {episode + 1}/{config.EPISODES} | PSR: {metrics['psr']:.2%} | SAP: {metrics['sap']:.2%} | Learning Reward: {learning_reward:.4f}")

    # --- Section 3: Finalization ---
    print(f"--- Training finished in {time.time() - start_time:.2f} seconds. ---")
    plot_training_dashboard(history, output_path)
    return agent
