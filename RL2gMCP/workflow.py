# RL2gMCP/workflow.py
# Author: Xin-Wei Huang
# Updated: 2026-06-10

"""
High-level workflow API for RL-based graphical MCP optimization.

This module provides app-friendly entry points with clear inputs and outputs:

- RLOptimizationConfig: user-facing configuration.
- run_rl_optimization: trains an RL agent and returns the learned graph.
- run_monte_carlo_evaluation: evaluates a graph with Monte Carlo simulation.
- plot_training_history / plot_graph: lightweight visualization helpers.

The functions here intentionally avoid FNN or other benchmark methods. They are
designed to be called from a local script, Streamlit app, Shiny for Python app,
Dash app, or R Shiny via reticulate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any, Callable
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.distributions import kl_divergence

from .agent import Agent
from .gMCP import gMCP
from .training import calculate_learning_reward


ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(slots=True)
class RLOptimizationConfig:
    """
    User-facing configuration for RL graph optimization.

    Use None inside alpha_weight_fix and t_fix to mark values that should be
    learned by the RL agent. The workflow converts those values to NaN
    internally for compatibility with the lower-level action processor.
    """

    # Scenario settings
    marginal_power: list[float]
    power_range: float
    reward_weights: list[float]

    # Clinical objective / reward settings
    primary_endpoints: list[int]
    primary_rule: str = "PRIMARY"
    reward_type: str = "psr"
    psr_threshold: float = 0.90
    psr_penalty: float = 10.0
    w_psr: float = 10.0
    w_sap: float = 1.0
    lexi_m: float = 100.0

    # P-value simulation settings
    corr_type: str | None = "AR1"
    corr_rho: float | None = 0.6
    alpha0: float = 0.05

    # RL training settings
    episodes: int = 4000
    batch_size: int = 128
    learning_rate: float = 1e-4
    baseline_decay: float = 0.99
    entropy_coeff: float = 0.001
    entropy_coeff_final: float | None = None
    random_seed: int | None = 10001
    device: str | torch.device | None = None
    dtype: torch.dtype = torch.float32

    # Agent architecture
    agent_hidden_layers: list[int] | None = None
    agent_activation_fn: str = "relu"
    agent_use_layer_norm: bool = True
    agent_top_k: int | None = None
    agent_dropout_rate: float | None = 0.2

    # Optimizer
    optimizer_type: str = "AdamW"
    weight_decay: float = 1e-2

    # Optional constraints. None means "learn this value".
    alpha_weight_fix: list[float | None] | None = None
    t_fix: list[list[float | None]] | None = None

    def resolved_device(self) -> torch.device:
        if self.device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["device"] = str(self.resolved_device())
        data["dtype"] = str(self.dtype).replace("torch.", "")
        return data


@dataclass(slots=True)
class RLGraph:
    """Learned graphical MCP procedure."""

    alpha_weight: np.ndarray
    transition_matrix: np.ndarray
    alpha0: float

    @property
    def alpha(self) -> np.ndarray:
        return self.alpha_weight * self.alpha0

    def to_dataframes(self, hypothesis_names: list[str] | None = None) -> dict[str, pd.DataFrame]:
        m = len(self.alpha_weight)
        names = hypothesis_names or [f"H{i + 1}" for i in range(m)]
        alpha_df = pd.DataFrame(
            {
                "Hypothesis": names,
                "Alpha weight": self.alpha_weight,
                "Initial alpha": self.alpha,
            }
        )
        t_df = pd.DataFrame(self.transition_matrix, index=names, columns=names)
        return {"alpha": alpha_df, "transition_matrix": t_df}


@dataclass(slots=True)
class RLOptimizationResult:
    """Result returned by run_rl_optimization."""

    graph: RLGraph
    training_history: pd.DataFrame
    config: RLOptimizationConfig
    elapsed_seconds: float
    agent: Agent | None = None

    def to_dataframes(self, hypothesis_names: list[str] | None = None) -> dict[str, pd.DataFrame]:
        data = self.graph.to_dataframes(hypothesis_names)
        data["training_history"] = self.training_history.copy()
        return data


@dataclass(slots=True)
class RLEvaluationResult:
    """Result returned by run_monte_carlo_evaluation."""

    summary_table: pd.DataFrame
    marginal_powers: np.ndarray
    psr: float
    sap: float
    avg_power: float
    num_simulations: int
    elapsed_seconds: float
    config: RLOptimizationConfig


def _activation_from_name(name: str) -> type[nn.Module]:
    activation_map = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }
    key = name.lower()
    if key not in activation_map:
        valid = ", ".join(sorted(activation_map))
        raise ValueError(f"Unsupported activation function '{name}'. Use one of: {valid}.")
    return activation_map[key]


def _none_to_nan_array(values: list[Any] | None, shape: tuple[int, ...], name: str) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.array(values, dtype=object)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}.")
    return np.array([[np.nan if x is None else x for x in row] for row in arr], dtype=float) if len(shape) == 2 else np.array(
        [np.nan if x is None else x for x in arr], dtype=float
    )


def _validate_config(config: RLOptimizationConfig) -> int:
    m = len(config.marginal_power)
    if m < 2:
        raise ValueError("At least two hypotheses are required.")
    if len(config.reward_weights) != m:
        raise ValueError("reward_weights must have the same length as marginal_power.")
    if len(config.primary_endpoints) != m:
        raise ValueError("primary_endpoints must have the same length as marginal_power.")
    if config.power_range < 0:
        raise ValueError("power_range must be non-negative.")
    if config.alpha0 <= 0 or config.alpha0 >= 1:
        raise ValueError("alpha0 must be between 0 and 1.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.episodes <= 0:
        raise ValueError("episodes must be positive.")
    if config.reward_type not in {"psr", "lexicographic", "avg_power"}:
        raise ValueError("reward_type must be one of: 'psr', 'lexicographic', 'avg_power'.")
    if config.primary_rule not in {"PRIMARY", "CO_PRIMARY", "DUAL_PRIMARY"}:
        raise ValueError("primary_rule must be one of: 'PRIMARY', 'CO_PRIMARY', 'DUAL_PRIMARY'.")
    if config.corr_type not in {None, "AR1", "CS"}:
        raise ValueError("corr_type must be None, 'AR1', or 'CS'.")
    if config.corr_type is not None and config.corr_rho is None:
        raise ValueError("corr_rho is required when corr_type is 'AR1' or 'CS'.")

    alpha_fix = _none_to_nan_array(config.alpha_weight_fix, (m,), "alpha_weight_fix")
    if alpha_fix is not None:
        fixed_sum = np.nansum(alpha_fix)
        if fixed_sum > 1.0 + 1e-8:
            raise ValueError(f"Fixed alpha weights sum to {fixed_sum:.6f}; they cannot exceed 1.")

    t_fix = _none_to_nan_array(config.t_fix, (m, m), "t_fix")
    if t_fix is not None:
        t_no_diag = t_fix.copy()
        np.fill_diagonal(t_no_diag, np.nan)
        row_sums = np.nansum(t_no_diag, axis=1)
        if np.any(row_sums > 1.0 + 1e-8):
            bad_rows = np.where(row_sums > 1.0 + 1e-8)[0] + 1
            raise ValueError(f"Fixed transition weights exceed 1 in row(s): {bad_rows.tolist()}.")

    return m


def _runtime_config(config: RLOptimizationConfig) -> SimpleNamespace:
    hidden_layers = config.agent_hidden_layers if config.agent_hidden_layers is not None else [128, 64]
    return SimpleNamespace(
        MARGINAL_POWER=config.marginal_power,
        POWER_RANGE=config.power_range,
        REWARD_WEIGHTS=config.reward_weights,
        REWARD_TYPE=config.reward_type,
        PRIMARY_ENDPOINTS=config.primary_endpoints,
        PRIMARY_RULE=config.primary_rule,
        PSR_THRESHOLD=config.psr_threshold,
        PSR_PENALTY=config.psr_penalty,
        W_PSR=config.w_psr,
        W_SAP=config.w_sap,
        LEXI_M=config.lexi_m,
        CORR_TYPE=config.corr_type,
        CORR_RHO=config.corr_rho,
        ALPHA_0=config.alpha0,
        LEARNING_RATE=config.learning_rate,
        EPISODES=config.episodes,
        BATCH_SIZE=config.batch_size,
        BASELINE_DECAY=config.baseline_decay,
        DTYPE=config.dtype,
        ENTROPY_COEFF=config.entropy_coeff,
        ENTROPY_COEFF_FINAL=config.entropy_coeff_final,
        RANDOM_SEED=config.random_seed,
        DEVICE=config.resolved_device(),
        AGENT_HIDDEN_LAYERS=hidden_layers,
        AGENT_ACTIVATION_FN=_activation_from_name(config.agent_activation_fn),
        AGENT_USE_LAYER_NORM=config.agent_use_layer_norm,
        AGENT_TOP_K=config.agent_top_k,
        AGENT_DROPOUT_RATE=config.agent_dropout_rate,
        OPTIMIZER_TYPE=config.optimizer_type,
        WEIGHT_DECAY=config.weight_decay,
        ALPHA_WEIGHT_FIX=_none_to_nan_array(config.alpha_weight_fix, (len(config.marginal_power),), "alpha_weight_fix"),
        T_FIX=_none_to_nan_array(config.t_fix, (len(config.marginal_power), len(config.marginal_power)), "t_fix"),
    )


def _learning_reward_from_summary(config: RLOptimizationConfig, psr: float, sap: float, avg_power: float) -> float:
    if config.reward_type == "psr":
        penalty = config.psr_penalty if psr < config.psr_threshold else 0.0
        return config.w_psr * psr + config.w_sap * sap - penalty
    if config.reward_type == "lexicographic":
        return config.lexi_m * min(psr, config.psr_threshold) + sap
    if config.reward_type == "avg_power":
        return avg_power
    raise ValueError(f"Unsupported reward_type: '{config.reward_type}'.")


def run_rl_optimization(
    config: RLOptimizationConfig,
    *,
    progress_callback: ProgressCallback | None = None,
    progress_gap: int = 100,
    return_agent: bool = True,
) -> RLOptimizationResult:
    """
    Train an RL agent and return the optimized graphical MCP procedure.

    Args:
        config: User-facing optimization configuration.
        progress_callback: Optional callback receiving a dictionary with episode
            number, metrics, loss, entropy, and elapsed seconds. This is useful
            for web apps that need to update a progress display.
        progress_gap: Callback/reporting interval in episodes.
        return_agent: If False, omit the trained PyTorch agent from the returned
            result to make the object lighter and easier to serialize.

    Returns:
        RLOptimizationResult containing the learned graph and training history.
    """
    start_time = time.time()
    m = _validate_config(config)
    cfg = _runtime_config(config)
    device = cfg.DEVICE

    if cfg.RANDOM_SEED is not None:
        np.random.seed(cfg.RANDOM_SEED)
        torch.manual_seed(cfg.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.RANDOM_SEED)

    agent = Agent(
        m=m,
        hidden_layers=cfg.AGENT_HIDDEN_LAYERS,
        activation_fn=cfg.AGENT_ACTIVATION_FN,
        use_layer_norm=cfg.AGENT_USE_LAYER_NORM,
        top_k=cfg.AGENT_TOP_K,
        dropout_rate=cfg.AGENT_DROPOUT_RATE,
    ).to(device)

    optimizer_name = cfg.OPTIMIZER_TYPE.lower()
    if optimizer_name == "adamw":
        optimizer = optim.AdamW(agent.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    elif optimizer_name == "adam":
        optimizer = optim.Adam(agent.parameters(), lr=cfg.LEARNING_RATE)
    else:
        raise ValueError("optimizer_type must be 'AdamW' or 'Adam'.")

    W = torch.tensor(cfg.REWARD_WEIGHTS, dtype=cfg.DTYPE, device=device)
    W_sum = torch.sum(W)
    W_scaled = W / W_sum if W_sum > 0 else W
    primary_mask = torch.tensor(cfg.PRIMARY_ENDPOINTS, dtype=cfg.DTYPE, device=device)
    cov_matrix = gMCP.create_covariance_matrix(m=m, corr_type=cfg.CORR_TYPE, rho=cfg.CORR_RHO)
    alpha_weight_fix_tensor = (
        torch.tensor(cfg.ALPHA_WEIGHT_FIX, dtype=cfg.DTYPE, device=device) if cfg.ALPHA_WEIGHT_FIX is not None else None
    )
    t_fix_tensor = torch.tensor(cfg.T_FIX, dtype=cfg.DTYPE, device=device) if cfg.T_FIX is not None else None

    history: list[dict[str, float | int]] = []
    reward_baseline = 0.0
    old_alpha_dist = None
    old_t_dists = None

    for episode in range(cfg.EPISODES):
        agent.train()

        with torch.no_grad():
            x = agent.base_layer(torch.ones(1, 1, device=device))
            alpha_logits = agent.alpha_head(x).squeeze(0)
            t_logits = agent.T_head(x).view(m, m)

        current_alpha_dist = torch.distributions.Dirichlet(F.softplus(alpha_logits) + 1e-6)
        current_t_dists = [torch.distributions.Dirichlet(F.softplus(t_logits[i]) + 1e-6) for i in range(m)]
        kl_div = 0.0
        if old_alpha_dist is not None and old_t_dists is not None:
            kl_alpha = kl_divergence(current_alpha_dist, old_alpha_dist).item()
            kl_t = sum(kl_divergence(current_t_dists[i], old_t_dists[i]).item() for i in range(m))
            kl_div = kl_alpha + kl_t
        old_alpha_dist = current_alpha_dist
        old_t_dists = current_t_dists

        alpha_weight, T, log_prob, entropy = agent(alpha_weight_fix=alpha_weight_fix_tensor, T_fix=t_fix_tensor)

        p_value_batch = gMCP.generate_p_values(
            batch_size=cfg.BATCH_SIZE,
            marginal_power=cfg.MARGINAL_POWER,
            power_range=cfg.POWER_RANGE,
            cov_matrix=cov_matrix,
            sim_alpha=cfg.ALPHA_0,
        ).to(dtype=cfg.DTYPE, device=device)
        decisions_batch = torch.stack(
            [gMCP.graphTest(alpha_weight.detach(), T.detach(), p, cfg.ALPHA_0) for p in p_value_batch]
        )

        learning_reward, metrics = calculate_learning_reward(cfg, decisions_batch, W_scaled, primary_mask, W, device)

        if episode == 0:
            reward_baseline = learning_reward
        else:
            reward_baseline = cfg.BASELINE_DECAY * reward_baseline + (1 - cfg.BASELINE_DECAY) * learning_reward
        advantage = learning_reward - reward_baseline

        entropy_coeff_now = cfg.ENTROPY_COEFF
        if cfg.ENTROPY_COEFF_FINAL is not None:
            frac = episode / max(cfg.EPISODES - 1, 1)
            entropy_coeff_now = cfg.ENTROPY_COEFF * (1 - frac) + cfg.ENTROPY_COEFF_FINAL * frac

        loss = -log_prob * advantage - entropy_coeff_now * entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        row = {
            "episode": episode + 1,
            "learning_reward": float(learning_reward),
            "avg_reward": float(metrics["avg_reward"]),
            "psr": float(metrics["psr"]),
            "sap": float(metrics["sap"]),
            "entropy": float(entropy.item()),
            "kl_divergence": float(kl_div),
            "loss": float(loss.item()),
            "elapsed_seconds": float(time.time() - start_time),
        }
        history.append(row)

        should_report = progress_callback is not None and (
            episode == 0
            or (episode + 1) == cfg.EPISODES
            or (progress_gap > 0 and (episode + 1) % progress_gap == 0)
        )
        if should_report and progress_callback is not None:
            progress_callback(row.copy())

    agent.eval()
    with torch.no_grad():
        final_alpha_weight, final_t, _, _ = agent(
            deterministic=True,
            alpha_weight_fix=alpha_weight_fix_tensor,
            T_fix=t_fix_tensor,
        )

    graph = RLGraph(
        alpha_weight=final_alpha_weight.cpu().numpy(),
        transition_matrix=final_t.cpu().numpy(),
        alpha0=cfg.ALPHA_0,
    )
    return RLOptimizationResult(
        graph=graph,
        training_history=pd.DataFrame(history),
        config=config,
        elapsed_seconds=time.time() - start_time,
        agent=agent if return_agent else None,
    )


def run_monte_carlo_evaluation(
    graph: RLGraph,
    config: RLOptimizationConfig,
    *,
    num_simulations: int = 10000,
    eval_batch_size: int = 2048,
    random_seed: int | None = None,
    hypothesis_names: list[str] | None = None,
) -> RLEvaluationResult:
    """
    Evaluate a learned graph by Monte Carlo simulation.

    Args:
        graph: Graph returned by run_rl_optimization, or a manually created RLGraph.
        config: Scenario and objective configuration.
        num_simulations: Number of Monte Carlo p-value vectors to simulate.
        eval_batch_size: Batch size used during evaluation.
        random_seed: Optional seed for reproducible evaluation. If omitted, uses
            config.random_seed.
        hypothesis_names: Optional names used in the returned summary table.

    Returns:
        RLEvaluationResult with a one-row summary table and scalar metrics.
    """
    start_time = time.time()
    m = _validate_config(config)
    if len(graph.alpha_weight) != m or graph.transition_matrix.shape != (m, m):
        raise ValueError("graph dimensions must match config.marginal_power.")
    if num_simulations <= 0:
        raise ValueError("num_simulations must be positive.")
    if eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be positive.")

    cfg = _runtime_config(config)
    device = cfg.DEVICE
    seed = config.random_seed if random_seed is None else random_seed
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    W = torch.tensor(cfg.REWARD_WEIGHTS, dtype=cfg.DTYPE, device=device)
    W_sum = torch.sum(W)
    W_scaled = W / W_sum if W_sum > 0 else W
    primary_mask = torch.tensor(cfg.PRIMARY_ENDPOINTS, dtype=cfg.DTYPE, device=device)
    cov_matrix = gMCP.create_covariance_matrix(m=m, corr_type=cfg.CORR_TYPE, rho=cfg.CORR_RHO)

    alpha_weight_t = torch.tensor(graph.alpha_weight, dtype=cfg.DTYPE, device=device)
    t_t = torch.tensor(graph.transition_matrix, dtype=cfg.DTYPE, device=device)

    total_psr = 0.0
    total_sap = 0.0
    total_power = 0.0
    marginal_rejections = np.zeros(m)

    with torch.no_grad():
        for i in range(0, num_simulations, eval_batch_size):
            current_batch_size = min(eval_batch_size, num_simulations - i)
            p_value_batch = gMCP.generate_p_values(
                batch_size=current_batch_size,
                marginal_power=cfg.MARGINAL_POWER,
                power_range=cfg.POWER_RANGE,
                cov_matrix=cov_matrix,
                sim_alpha=cfg.ALPHA_0,
            ).to(dtype=cfg.DTYPE, device=device)

            decisions = torch.stack([gMCP.graphTest(alpha_weight_t, t_t, p, cfg.ALPHA_0) for p in p_value_batch])
            _, metrics = calculate_learning_reward(cfg, decisions, W_scaled, primary_mask, W, device)
            total_psr += metrics["psr"] * current_batch_size
            total_sap += metrics["sap"] * current_batch_size
            total_power += (decisions.float() @ W_scaled).sum().item()
            marginal_rejections += torch.sum(decisions, dim=0).cpu().numpy()

    psr = total_psr / num_simulations
    sap = total_sap / num_simulations
    avg_power = total_power / num_simulations
    learning_reward = _learning_reward_from_summary(config, psr, sap, avg_power)
    marginal_powers = marginal_rejections / num_simulations

    names = hypothesis_names or [f"H{i + 1}" for i in range(m)]
    row = {
        "Graph": "RL Agent",
        "PSR": psr,
        "SAP": sap,
        "Avg Power": avg_power,
        "Learning Reward": learning_reward,
    }
    for name, value in zip(names, marginal_powers):
        row[name] = value

    return RLEvaluationResult(
        summary_table=pd.DataFrame([row]),
        marginal_powers=marginal_powers,
        psr=psr,
        sap=sap,
        avg_power=avg_power,
        num_simulations=num_simulations,
        elapsed_seconds=time.time() - start_time,
        config=config,
    )


def plot_training_history(training_history: pd.DataFrame) -> plt.Figure:
    """Create a compact matplotlib dashboard from run_rl_optimization history."""
    required = {"episode", "learning_reward", "psr", "sap", "entropy", "loss"}
    missing = required - set(training_history.columns)
    if missing:
        raise ValueError(f"training_history is missing required columns: {sorted(missing)}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes_flat = axes.ravel()
    plot_specs = [
        ("learning_reward", "Learning Reward"),
        ("psr", "Primary Success Rate"),
        ("sap", "Secondary Average Power"),
        ("entropy", "Policy Entropy"),
        ("loss", "Loss"),
        ("kl_divergence", "Policy KL Divergence"),
    ]
    for ax, (column, title) in zip(axes_flat, plot_specs):
        if column not in training_history:
            ax.axis("off")
            continue
        ax.plot(training_history["episode"], training_history[column], linewidth=1.3)
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_graph(graph: RLGraph, hypothesis_names: list[str] | None = None) -> plt.Figure:
    """
    Plot the learned graph with matplotlib.

    This avoids an R dependency. For publication-style gMCPLite plots, use
    translateR.get_final_graph_in_R or export the alpha/T tables.
    """
    m = len(graph.alpha_weight)
    names = hypothesis_names or [f"H{i + 1}" for i in range(m)]
    theta = np.linspace(0, 2 * np.pi, m, endpoint=False)
    positions = np.column_stack([np.cos(theta), np.sin(theta)])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.axis("off")

    max_edge = np.max(graph.transition_matrix) if graph.transition_matrix.size else 0.0
    for i in range(m):
        for j in range(m):
            weight = graph.transition_matrix[i, j]
            if i == j or weight <= 1e-8:
                continue
            start = positions[i]
            end = positions[j]
            width = 0.8 + 3.0 * (weight / max_edge if max_edge > 0 else 0)
            ax.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops={
                    "arrowstyle": "->",
                    "lw": width,
                    "color": "#5b6770",
                    "alpha": 0.65,
                    "shrinkA": 24,
                    "shrinkB": 24,
                    "connectionstyle": "arc3,rad=0.12",
                },
            )
            mid = (start + end) / 2
            ax.text(mid[0], mid[1], f"{weight:.2f}", ha="center", va="center", fontsize=9, color="#2f3b45")

    for idx, (x, y) in enumerate(positions):
        circle = plt.Circle((x, y), 0.18, facecolor="#f7f9fb", edgecolor="#1f77b4", linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y + 0.035, names[idx], ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(x, y - 0.075, f"a={graph.alpha[idx]:.3f}", ha="center", va="center", fontsize=8)

    ax.set_title("Optimized Graphical MCP Procedure")
    fig.tight_layout()
    return fig


__all__ = [
    "RLOptimizationConfig",
    "RLGraph",
    "RLOptimizationResult",
    "RLEvaluationResult",
    "run_rl_optimization",
    "run_monte_carlo_evaluation",
    "plot_training_history",
    "plot_graph",
]
