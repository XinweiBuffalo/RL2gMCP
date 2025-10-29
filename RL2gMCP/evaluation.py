# RL2gMCP/evaluation.py
# Author: Xin-Wei Huang
# Updated: 2025-07-16

"""
This file provides functions for the final evaluation and benchmarking phase.
It takes a trained agent and performs a large-scale Monte Carlo simulation
to assess its performance against other methods.

Functions:
- get_final_graph: Extracts the final, deterministic policy from a trained agent.
- evaluate_policy: The main evaluation function that runs simulations, calculates
                   a suite of performance metrics (PSR, SAP, Avg Power), and
                   prints/saves a comprehensive report.
"""

import torch
import numpy as np
import time
import pandas as pd
from .gMCP import gMCP
from .utils import Hyperparameters
from .agent import Agent
from .training import calculate_primary_success_rate, calculate_secondary_power


def get_final_graph(agent: Agent, alpha_weight_fix: torch.Tensor | None = None, T_fix: torch.Tensor | None = None):
    """
    Extracts the final, deterministic graph from a trained agent.

    Args:
        agent (Agent): The trained RL agent.
        alpha_weight_fix (torch.Tensor | None, optional): Constraints for alpha_weight.
        T_fix (torch.Tensor | None, optional): Constraints for T matrix.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - alpha_weight (torch.Tensor): The final weight vector.
            - T (torch.Tensor): The final transition matrix.
    """
    agent.eval()  # Set agent to evaluation mode
    with torch.no_grad():
        # The agent's forward pass generates the deterministic graph
        # by taking the mean of the policy distributions.
        alpha_weight, T, _, _ = agent(
            deterministic = True,
            alpha_weight_fix = alpha_weight_fix,
            T_fix = T_fix
        )
    return alpha_weight.cpu().detach(), T.cpu().detach()


def evaluate_policy(
        agent: Agent,
        config: Hyperparameters,
        num_simulations: int = 10000,
        custom_graphs: list = None,
        output_path: str = None,
        setting_name: str = "None"
):
    """
    Performs a large-scale evaluation and generates a performance report.
    """
    # --- Section 1: Initialization and Setup ---
    device = config.DEVICE

    if config.RANDOM_SEED is not None:
        print(f"Using fixed random seed for evaluation: {config.RANDOM_SEED}")
        np.random.seed(config.RANDOM_SEED)
        torch.manual_seed(config.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.RANDOM_SEED)

    m = len(config.REWARD_WEIGHTS)
    print(f"\n--- Starting Final Evaluation for setting: '{setting_name}' ---")

    W = torch.tensor(config.REWARD_WEIGHTS, dtype = config.DTYPE, device = device)
    W_sum = torch.sum(W)
    W_scaled = W / W_sum if W_sum > 0 else W
    primary_mask = torch.tensor(config.PRIMARY_ENDPOINTS, dtype = config.DTYPE, device = device)

    alpha_weight_fix_tensor = torch.tensor(config.ALPHA_WEIGHT_FIX, dtype = config.DTYPE,
                                           device = device) if config.ALPHA_WEIGHT_FIX is not None else None
    T_fix_tensor = torch.tensor(config.T_FIX, dtype = config.DTYPE,
                                device = device) if config.T_FIX is not None else None
    cov_matrix_eval = gMCP.create_covariance_matrix(m = m, corr_type = config.CORR_TYPE, rho = config.CORR_RHO)

    start_time = time.time()

    # This function call's inputs are the trained agent and any constraints.
    # It returns the final, deterministic graph for evaluation.
    final_alpha_weight, final_T = get_final_graph(agent, alpha_weight_fix_tensor, T_fix_tensor)

    # --- Section 2: Monte Carlo Simulation Loop ---
    method_names = ['RL Agent']
    if custom_graphs:
        for graph_spec in custom_graphs:
            method_names.append(graph_spec['name'])

    total_metrics = {name: {'psr': 0.0, 'sap': 0.0, 'power': 0.0, 'marginal_rejections': np.zeros(m)} for name in
                     method_names}

    eval_batch_size = 2048
    with torch.no_grad():
        for i in range(0, num_simulations, eval_batch_size):
            current_batch_size = min(eval_batch_size, num_simulations - i)

            # This function call generates a batch of p-values based on config.
            # It returns a tensor of shape (batch_size, m).
            p_value_batch = gMCP.generate_p_values(batch_size = current_batch_size,
                                                   marginal_power = config.MARGINAL_POWER,
                                                   power_range = config.POWER_RANGE, cov_matrix = cov_matrix_eval,
                                                   sim_alpha = config.ALPHA_0).to(dtype = config.DTYPE, device = device)

            # --- Subsection 2.1: Evaluate RL Agent's Graph ---
            decisions_rl = torch.stack(
                [gMCP.graphTest(final_alpha_weight.to(device), final_T.to(device), p, config.ALPHA_0) for p in
                 p_value_batch])
            total_metrics['RL Agent']['psr'] += calculate_primary_success_rate(decisions_rl, primary_mask,
                                                                               config.PRIMARY_RULE,
                                                                               device) * current_batch_size
            total_metrics['RL Agent']['sap'] += calculate_secondary_power(decisions_rl, primary_mask, W,
                                                                          device) * current_batch_size
            total_metrics['RL Agent']['power'] += (decisions_rl.float() @ W_scaled.to(device)).sum().item()
            total_metrics['RL Agent']['marginal_rejections'] += torch.sum(decisions_rl, dim = 0).cpu().numpy()

            # --- Subsection 2.2: Evaluate Custom Benchmark Graphs ---
            if custom_graphs:
                for graph_spec in custom_graphs:
                    name = graph_spec['name']
                    decisions_custom = torch.stack([gMCP.graphTest(graph_spec['alpha_weight'].to(device),
                                                                   graph_spec['T'].to(device), p, config.ALPHA_0) for p
                                                    in p_value_batch])
                    total_metrics[name]['psr'] += calculate_primary_success_rate(decisions_custom, primary_mask,
                                                                                 config.PRIMARY_RULE,
                                                                                 device) * current_batch_size
                    total_metrics[name]['sap'] += calculate_secondary_power(decisions_custom, primary_mask, W,
                                                                            device) * current_batch_size
                    total_metrics[name]['power'] += (decisions_custom.float() @ W_scaled.to(device)).sum().item()
                    total_metrics[name]['marginal_rejections'] += torch.sum(decisions_custom, dim = 0).cpu().numpy()

    # --- Section 3: Process and Report Results ---
    avg_metrics = {
        name: {
            'psr': totals['psr'] / num_simulations,
            'sap': totals['sap'] / num_simulations,
            'power': totals['power'] / num_simulations,
            'marginal_powers': totals['marginal_rejections'] / num_simulations
        } for name, totals in total_metrics.items()
    }

    print(f"Evaluation finished in {time.time() - start_time:.2f} seconds.")

    # --- Subsection 3.1: Create DataFrame for Saving ---
    report_data = []
    for name, metrics in avg_metrics.items():
        data_row = {'Graph': name, 'PSR': metrics['psr'], 'SAP': metrics['sap'], 'Avg Power': metrics['power']}
        for i in range(m): data_row[f'H{i + 1}'] = metrics['marginal_powers'][i]
        report_data.append(data_row)
    df = pd.DataFrame(report_data).sort_values(by = 'Avg Power', ascending = False).reset_index(drop = True)

    # --- Subsection 3.2: Print Formatted Report to Console ---
    # --- Print to Console with New Format ---
    print("\n--- Performance Comparison Report ---")

    # Define column widths for neat alignment
    graph_width, psr_width, sap_width, power_width, h_width = 25, 22, 26, 17, 10
    separator = " | "

    # Construct the first header line with all metric names
    header1 = (f"{'Graph':<{graph_width}}{separator}{'Primary success rate':<{psr_width}}{separator}"
               f"{'Secondary average power':<{sap_width}}{separator}{'Average Power':<{power_width}}")
    for i in range(m): header1 += f"{separator}{f'H{i + 1}':<{h_width}}"
    print(header1)

    # Construct the second header line for scaled weights
    scaled_weights_label = "Scaled Weights"
    # Calculate width to span the first four columns
    header2_part1_width = graph_width + psr_width + sap_width + power_width + (len(separator) * 4)
    header2 = f"{scaled_weights_label:<{header2_part1_width}}"

    scaled_weights_list = W_scaled.cpu().numpy()
    for weight in scaled_weights_list:
        # Format scaled weights to 3 decimal places for consistency
        header2 += f"{separator}{weight:<{h_width}.3f}"
    print(header2)

    # Print the separator line
    print("-" * len(header1))

    # Print the data rows for each graph with the new formatting
    for _, row in df.iterrows():
        # This is the line with the main formatting changes
        row_str = (f"{row['Graph']:<{graph_width}}{separator}"
                   f"{row['PSR']:<{psr_width}.2%}{separator}"  # Keep as percentage
                   f"{row['SAP']:<{sap_width}.3f}{separator}"  # Change to 3 decimal places
                   f"{row['Avg Power']:<{power_width}.3f}")  # Change to 3 decimal places
        for i in range(m):
            # Change all marginal powers to 3 decimal places
            row_str += f"{separator}{row[f'H{i + 1}']:<{h_width}.3f}"
        print(row_str)
    print("-" * len(header1))

    # --- Subsection 3.3: Save Full Report to CSV ---
    if output_path:
        df_to_save = df.copy()
        df_to_save.insert(0, 'Setting', setting_name)
        try:
            df_to_save.to_csv(f"{output_path}.csv", index = False)
            print(f"\nSuccessfully saved detailed report.")
        except Exception as e:
            print(f"Error saving CSV file: {e}")
