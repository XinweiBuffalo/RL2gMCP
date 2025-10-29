# RL2gMCP/fnn_benchmark.py
# Author: Xin-Wei Huang (with FNN method based on Zhan, et al., 2022)
# Updated: 2025-07-24

"""
This module implements the FNN-based optimization framework described by
Zhan et al. (2022). This version uses a dedicated parameter for the number
of simulations used to generate FNN training labels, separate from the RL
agent's batch size.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from scipy.optimize import minimize, Bounds

from .gMCP import gMCP
from .utils import Hyperparameters
from .training import calculate_learning_reward


# --- Helper: Random Graph Generation (Paper's Method) ---

def _generate_random_constrained_graph(alpha_weight_fix, T_fix, m, device = 'cpu'):
    """Generates a single random graph that respects the given constraints."""
    alpha_weight = torch.zeros(m, device = device, dtype = torch.float32)
    if alpha_weight_fix is None:
        budget = 1.0
        for i in range(m - 1):
            val = torch.rand((), device = device, dtype = torch.float32).item() * budget
            alpha_weight[i] = val
            budget -= val
        alpha_weight[m - 1] = budget
    else:
        learn_mask = torch.isnan(alpha_weight_fix)
        fixed_values = torch.nan_to_num(alpha_weight_fix, nan = 0.0)
        budget = (1.0 - torch.sum(fixed_values)).item()
        alpha_weight = fixed_values.clone()
        learn_indices = torch.where(learn_mask)[0]
        for i in range(len(learn_indices) - 1):
            idx = learn_indices[i]
            val = torch.rand((), device = device, dtype = torch.float32).item() * budget
            alpha_weight[idx] = val
            budget -= val
        if len(learn_indices) > 0:
            alpha_weight[learn_indices[-1]] = budget

    T = torch.zeros(m, m, device = device, dtype = torch.float32)
    if T_fix is None:
        for i in range(m):
            row_budget = 1.0
            row_indices = list(range(m))
            row_indices.pop(i)
            for j in row_indices[:-1]:
                val = torch.rand((), device = device, dtype = torch.float32).item() * row_budget
                T[i, j] = val
                row_budget -= val
            T[i, row_indices[-1]] = row_budget
    else:
        for i in range(m):
            row_fix = T_fix[i]
            learn_mask = torch.isnan(row_fix)
            learn_mask[i] = False
            fixed_values = torch.nan_to_num(row_fix, nan = 0.0)
            fixed_values[i] = 0.0
            row_budget = (1.0 - torch.sum(fixed_values)).item()
            T[i, :] = fixed_values
            learn_indices = torch.where(learn_mask)[0]
            for j in range(len(learn_indices) - 1):
                idx = learn_indices[j]
                val = torch.rand((), device = device, dtype = torch.float32).item() * row_budget
                T[i, idx] = val
                row_budget -= val
            if len(learn_indices) > 0:
                T[i, learn_indices[-1]] = row_budget
    return alpha_weight, T


# --- FNN Surrogate Model Definition ---

class FNN_Surrogate_Model(nn.Module):
    """A Feed-Forward Network to act as a surrogate for the objective function."""

    def __init__(self, input_dim: int, hidden_layers: list, dropout_rate: float):
        super(FNN_Surrogate_Model, self).__init__()
        layers = []
        in_features = input_dim
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_size
        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# --- Main Orchestration Function ---

def find_optimal_graph_fnn(
        config: Hyperparameters,
        num_fnn_samples: int,
        fnn_epochs: int,
        cobyla_max_iter: int,
        fnn_label_sims: int
):
    """Finds the optimal graph using the full FNN pipeline from the paper."""
    m = len(config.MARGINAL_POWER)
    device = config.DEVICE

    alpha_fix_tensor = torch.tensor(config.ALPHA_WEIGHT_FIX, device = device,
                                    dtype = torch.float32) if config.ALPHA_WEIGHT_FIX is not None else None
    T_fix_tensor = torch.tensor(config.T_FIX, device = device,
                                dtype = torch.float32) if config.T_FIX is not None else None

    # --- Stage 1: Generate Data and Split ---
    print(f"Generating {num_fnn_samples} samples for FNN, using {fnn_label_sims} simulations per sample...")

    input_dim = m + m * m
    X_data = torch.zeros((num_fnn_samples, input_dim), device = device)
    Y_data = torch.zeros((num_fnn_samples, 1), device = device)

    start_time = time.time()
    for i in range(num_fnn_samples):
        alpha_w, T_m = _generate_random_constrained_graph(alpha_fix_tensor, T_fix_tensor, m, device)
        X_data[i] = torch.cat([alpha_w, T_m.flatten()])

        with torch.no_grad():
            # FIX: Use the dedicated fnn_label_sims parameter here
            p_value_batch = gMCP.generate_p_values(batch_size = fnn_label_sims, marginal_power = config.MARGINAL_POWER,
                                                   power_range = config.POWER_RANGE,
                                                   cov_matrix = gMCP.create_covariance_matrix(m = m,
                                                                                              corr_type = config.CORR_TYPE,
                                                                                              rho = config.CORR_RHO)).to(
                dtype = config.DTYPE, device = device)
            decisions_batch = torch.stack(
                [gMCP.graphTest(alpha_w.detach(), T_m.detach(), p, config.ALPHA_0) for p in p_value_batch])
            learning_reward, _ = calculate_learning_reward(config, decisions_batch,
                                                           torch.tensor(config.REWARD_WEIGHTS, device = device) / sum(
                                                               config.REWARD_WEIGHTS),
                                                           torch.tensor(config.PRIMARY_ENDPOINTS, device = device),
                                                           torch.tensor(config.REWARD_WEIGHTS, device = device), device)
            Y_data[i] = learning_reward

    print(f"Data generation finished in {time.time() - start_time:.2f} seconds.")

    shuffled_indices = torch.randperm(num_fnn_samples)
    split_idx = int(num_fnn_samples * 0.8)
    train_indices, val_indices = shuffled_indices[:split_idx], shuffled_indices[split_idx:]
    X_train, Y_train = X_data[train_indices], Y_data[train_indices]
    X_val, Y_val = X_data[val_indices], Y_data[val_indices]

    # --- Stage 2: FNN Model Selection ---
    print("\nStarting FNN model selection across 6 candidate architectures...")

    candidate_architectures = [
        {'layers': [30, 30], 'dropout': 0.3, 'name': 'L2_D0.3'},
        {'layers': [30, 30, 30], 'dropout': 0.3, 'name': 'L3_D0.3'},
        {'layers': [30, 30, 30, 30], 'dropout': 0.3, 'name': 'L4_D0.3'},
        {'layers': [30, 30], 'dropout': 0.0, 'name': 'L2_D0.0'},
        {'layers': [30, 30, 30], 'dropout': 0.0, 'name': 'L3_D0.0'},
        {'layers': [30, 30, 30, 30], 'dropout': 0.0, 'name': 'L4_D0.0'},
    ]
    best_fnn_model, best_val_mse = None, float('inf')
    loss_fn = nn.MSELoss()

    for arch in candidate_architectures:
        fnn_model = FNN_Surrogate_Model(input_dim = input_dim, hidden_layers = arch['layers'],
                                        dropout_rate = arch['dropout']).to(device)
        optimizer = optim.AdamW(fnn_model.parameters(), lr = config.LEARNING_RATE, weight_decay = config.WEIGHT_DECAY)
        for epoch in range(fnn_epochs):
            fnn_model.train()
            optimizer.zero_grad()
            loss = loss_fn(fnn_model(X_train), Y_train)
            loss.backward()
            optimizer.step()

        fnn_model.eval()
        with torch.no_grad():
            val_mse = loss_fn(fnn_model(X_val), Y_val).item()
        if val_mse < best_val_mse:
            best_val_mse, best_fnn_model = val_mse, fnn_model
            print(f"    *** New best model found: {arch['name']} with Val MSE: {val_mse:.6f} ***")

    # ... (Rest of the function remains the same, including SLSQP and COBYLA) ...

    print(f"\nBest FNN model selected with Validation MSE: {best_val_mse:.6f}")
    print("\nOptimizing on the best FNN surrogate with SLSQP...")
    best_fnn_model.eval()

    def fnn_objective_and_grad(params_np):
        params_torch = torch.from_numpy(params_np).float().to(device).requires_grad_()
        obj = -best_fnn_model(params_torch).sum()
        obj.backward()
        return obj.item(), params_torch.grad.cpu().numpy()

    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x[:m]) - 1.0}]
    for i in range(m):
        constraints.append({'type': 'eq', 'fun': lambda x, i = i: np.sum(x[m + i * m: m + (i + 1) * m]) - 1.0})
        constraints.append({'type': 'eq', 'fun': lambda x, i = i: x[m + i * m + i]})

    if alpha_fix_tensor is not None:
        for i, val in enumerate(alpha_fix_tensor):
            if not torch.isnan(val):
                constraints.append({'type': 'eq', 'fun': lambda x, i = i, val = val: x[i] - val.item()})
    if T_fix_tensor is not None:
        for i in range(m):
            for j in range(m):
                val = T_fix_tensor[i, j]
                if not torch.isnan(val):
                    constraints.append(
                        {'type': 'eq', 'fun': lambda x, i = i, j = j, val = val: x[m + i * m + j] - val.item()})

    bounds = Bounds(lb = 0.0, ub = 1.0)
    initial_params = X_data[torch.argmax(Y_data)].cpu().numpy()

    res_slsqp = minimize(lambda x: fnn_objective_and_grad(x)[0], initial_params, method = 'SLSQP',
                         jac = lambda x: fnn_objective_and_grad(x)[1], bounds = bounds, constraints = constraints,
                         options = {'disp': True, 'maxiter': 200})
    params_after_slsqp = res_slsqp.x

    print("Finished SLSQP optimization. Candidate solution found.")
    print("\nFine-tuning candidate solution with COBYLA...")

    def mc_objective(params_np):
        alpha_w = torch.from_numpy(params_np[:m]).float()
        T_m = torch.from_numpy(params_np[m:]).float().view(m, m)
        alpha_w = torch.clamp(alpha_w / alpha_w.sum(), 0, 1) if alpha_w.sum() > 0 else alpha_w
        T_m.fill_diagonal_(0)
        row_sums = T_m.sum(dim = 1, keepdim = True)
        T_m = torch.clamp(T_m / row_sums.clamp(min = 1e-9), 0, 1)

        with torch.no_grad():
            p_value_batch = gMCP.generate_p_values(batch_size = config.BATCH_SIZE,
                                                   marginal_power = config.MARGINAL_POWER,
                                                   power_range = config.POWER_RANGE,
                                                   cov_matrix = gMCP.create_covariance_matrix(m = m,
                                                                                              corr_type = config.CORR_TYPE,
                                                                                              rho = config.CORR_RHO)).to(
                dtype = config.DTYPE, device = device)
            decisions_batch = torch.stack(
                [gMCP.graphTest(alpha_w.to(device), T_m.to(device), p, config.ALPHA_0) for p in p_value_batch])
            learning_reward, _ = calculate_learning_reward(config, decisions_batch,
                                                           torch.tensor(config.REWARD_WEIGHTS, device = device) / sum(
                                                               config.REWARD_WEIGHTS),
                                                           torch.tensor(config.PRIMARY_ENDPOINTS, device = device),
                                                           torch.tensor(config.REWARD_WEIGHTS, device = device), device)
        return -learning_reward

    res_cobyla = minimize(mc_objective, params_after_slsqp, method = 'COBYLA',
                          options = {'disp': True, 'maxiter': cobyla_max_iter})
    final_params_np = res_cobyla.x

    alpha_final = torch.from_numpy(final_params_np[:m]).float()
    T_final = torch.from_numpy(final_params_np[m:]).float().view(m, m)

    alpha_final = torch.clamp(alpha_final, min = 0.0)
    T_final = torch.clamp(T_final, min = 0.0)

    alpha_final = alpha_final / alpha_final.sum()
    T_final.fill_diagonal_(0)
    row_sums = T_final.sum(dim = 1, keepdim = True)
    T_final = T_final / row_sums.clamp(min = 1e-9)

    print("\nFNN-based optimization with fine-tuning complete.")
    return alpha_final.cpu(), T_final.cpu()
