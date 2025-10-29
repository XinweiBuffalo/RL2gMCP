# RL2gMCP/action_processor.py
# Author: Xin-Wei Huang
# Updated: 2025-07-16

"""
This file contains modular functions for processing the raw outputs (logits)
from the agent's neural network. It handles the complex logic of applying
constraints, managing budgets, and interacting with sparsity settings to
generate valid, structured actions (alpha_weight and T matrix).

This separation of concerns keeps the agent's forward pass clean and focused
on neural network computations.

Functions:
- process_alpha_weight: Generates the alpha_weight vector.
- process_T_matrix: Generates the T matrix.
"""

import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet


def process_alpha_weight(
        alpha_logits: torch.Tensor,
        alpha_weight_fix: torch.Tensor | None,
        deterministic: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Processes logits to generate the alpha_weight vector, respecting constraints.

    Args:
        alpha_logits (torch.Tensor): Raw output from the agent's alpha_head. Shape: (m,).
        alpha_weight_fix (torch.Tensor | None): A tensor with fixed values or np.nan.
            Example: [0.5, nan, nan, 0.1]
        deterministic (bool): If True, returns the mean of the distribution. If False, samples.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - final_alpha_weight (torch.Tensor): The final weight vector. Shape: (m,).
            - log_prob (torch.Tensor): The log probability of the sampled action.
            - entropy (torch.Tensor): The entropy of the policy distribution.
    """
    device = alpha_logits.device
    log_prob = torch.tensor(0.0, device = device)
    entropy = torch.tensor(0.0, device = device)

    # --- Case 1: No constraints, learn the full vector ---
    if alpha_weight_fix is None:
        lambda_alpha_weight = F.softplus(alpha_logits) + 1e-6
        dist = Dirichlet(lambda_alpha_weight)
        sample = dist.mean if deterministic else dist.sample()
        if not deterministic:
            log_prob = dist.log_prob(sample)
            entropy = dist.entropy()
        return sample, log_prob, entropy

    # --- Case 2: Constraints are present ---
    else:
        # --- Section 1: Calculate Budget for Learnable Part ---
        learn_mask = torch.isnan(alpha_weight_fix)
        fixed_values = torch.nan_to_num(alpha_weight_fix, nan = 0.0)
        fixed_sum = torch.sum(fixed_values)
        if fixed_sum > 1.0 + 1e-6:
            raise ValueError(f"Sum of fixed alpha weights ({fixed_sum}) cannot exceed 1.0")
        budget = 1.0 - fixed_sum

        final_weight = fixed_values.clone()

        # --- Section 2: Learn the Proportions for the Budget ---
        if torch.sum(learn_mask) > 0:
            learnable_logits = alpha_logits[learn_mask]
            lambda_learnable = F.softplus(learnable_logits) + 1e-6
            dist = Dirichlet(lambda_learnable)
            props = dist.mean if deterministic else dist.sample()

            if not deterministic:
                log_prob = dist.log_prob(props)
                entropy = dist.entropy()

            # --- Section 3: Combine Fixed and Learned Parts ---
            final_weight[learn_mask] = props * budget

        return final_weight, log_prob, entropy


def process_T_matrix(
        T_logits: torch.Tensor,
        T_fix: torch.Tensor | None,
        top_k: int | None,
        deterministic: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Processes logits row-by-row to generate the T matrix.

    Args:
        T_logits (torch.Tensor): Raw output from the agent's T_head. Shape: (m, m).
        T_fix (torch.Tensor | None): A tensor with fixed values or np.nan for T.
        top_k (int | None): The sparsity parameter.
        deterministic (bool): If True, returns the mean. If False, samples.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - final_T_matrix_raw (torch.Tensor): The raw T matrix before final normalization.
            - log_prob (torch.Tensor): The total log probability.
            - entropy (torch.Tensor): The total entropy.
    """
    m = T_logits.shape[0]
    device = T_logits.device

    T_sample_raw_rows = []
    log_prob_rows = []
    entropy_rows = []

    # Process each row of the T matrix independently
    for i in range(m):
        T_fix_row = T_fix[i] if T_fix is not None else None
        has_constraints = T_fix_row is not None and not torch.all(torch.isnan(T_fix_row))

        # --- Case 1: The row has user-defined constraints (ignore top_k) ---
        if has_constraints:
            learn_mask = torch.isnan(T_fix_row)
            learn_mask[i] = False  # The diagonal is never learnable
            fixed_values = torch.nan_to_num(T_fix_row, nan = 0.0)
            fixed_values[i] = 0.0  # Enforce zero diagonal on fixed values

            fixed_sum = torch.sum(fixed_values)
            if fixed_sum > 1.0 + 1e-6:
                raise ValueError(f"Sum of fixed T values in row {i} ({fixed_sum}) cannot exceed 1.0")

            budget = 1.0 - fixed_sum
            final_row = fixed_values.clone()

            if torch.sum(learn_mask) > 0:
                learnable_logits = T_logits[i, learn_mask]
                lambda_learnable = F.softplus(learnable_logits) + 1e-6
                dist = Dirichlet(lambda_learnable)
                props = dist.mean if deterministic else dist.sample()
                if not deterministic:
                    log_prob_rows.append(dist.log_prob(props))
                    entropy_rows.append(dist.entropy())
                final_row[learn_mask] = props * budget

            T_sample_raw_rows.append(final_row)

        # --- Case 2: The row is fully learned by the agent (respect top_k) ---
        else:
            row_logits = T_logits[i].clone()
            row_logits[i] = -torch.inf  # Exclude diagonal from the start

            use_sparsity = top_k is not None and top_k > 0 and top_k < m - 1
            if use_sparsity:
                top_k_logits, top_k_indices = torch.topk(row_logits, top_k)
                lambda_T_row = F.softplus(top_k_logits) + 1e-6
                dist = Dirichlet(lambda_T_row)
                k_sample = dist.mean if deterministic else dist.sample()
                if not deterministic:
                    log_prob_rows.append(dist.log_prob(k_sample))
                    entropy_rows.append(dist.entropy())
                sparse_row = torch.zeros(m, device = device)
                sparse_row.scatter_(0, top_k_indices, k_sample)
                T_sample_raw_rows.append(sparse_row)
            else:  # Fully learned dense row
                lambda_T_row = F.softplus(row_logits) + 1e-6
                dist = Dirichlet(lambda_T_row)
                row_sample = dist.mean if deterministic else dist.sample()
                if not deterministic:
                    log_prob_rows.append(dist.log_prob(row_sample))
                    entropy_rows.append(dist.entropy())
                T_sample_raw_rows.append(row_sample)

    # --- Combine results from all rows ---
    T_sample_raw = torch.stack(T_sample_raw_rows)
    log_prob = torch.sum(torch.stack(log_prob_rows)) if log_prob_rows else torch.tensor(0.0, device = device)
    entropy = torch.sum(torch.stack(entropy_rows)) if entropy_rows else torch.tensor(0.0, device = device)

    return T_sample_raw, log_prob, entropy
