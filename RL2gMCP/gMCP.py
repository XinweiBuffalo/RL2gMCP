# RL2gMCP/gMCP.py
# Author: Xin-Wei Huang
# Updated: 2025-07-16

"""
This file contains the gMCP class, which provides a collection of static methods
for simulating and executing graphical multiple comparison procedures. It serves as
the core "environment" for the reinforcement learning agent.

Classes:
- gMCP: A utility class with static methods for:
    - Creating covariance matrices (`create_covariance_matrix`).
    - Generating p-values from statistical assumptions (`generate_p_values`).
    - Executing the sequential graphical test procedure (`graphTest`).
"""

import torch
import numpy as np
from scipy.stats import norm


class gMCP:
    """
    A utility class for graphical multiple comparison procedures.
    All methods are static and can be called directly using the class name,
    e.g., `gMCP.graphTest(...)`.
    """

    @staticmethod
    def create_covariance_matrix(
            m: int,
            corr_type: str | None = None,
            rho: float | None = None,
            sd_vec: np.ndarray = None
    ) -> np.ndarray:
        """
        Creates a covariance matrix with a specified correlation structure.

        Args:
            m (int): The dimension of the matrix (number of hypotheses).
            corr_type (str | None, optional): The correlation structure.
                Can be "AR1", "CS", or None. Defaults to None.
            rho (float | None, optional): The correlation coefficient. Required for
                "AR1" and "CS" structures. Defaults to None.
            sd_vec (np.ndarray, optional): A 1D array of standard deviations.
                If None, defaults to a vector of ones. Defaults to None.

        Returns:
            np.ndarray: An m x m NumPy array representing the covariance matrix.
                Example: A 2x2 CS matrix with rho=0.5 -> [[1.0, 0.5], [0.5, 1.0]]
        """
        # --- Section 1: Input Validation ---
        if corr_type is not None and rho is None:
            raise ValueError("A 'rho' value must be provided for 'AR1' or 'CS' correlation types.")

        # --- Section 2: Generate Correlation Matrix based on Type ---
        corr_matrix = None
        match corr_type:
            case None:
                corr_matrix = np.identity(m)
            case "AR1":
                indices = np.arange(m)
                abs_diff_matrix = np.abs(indices[:, np.newaxis] - indices)
                corr_matrix = rho ** abs_diff_matrix
            case "CS":
                corr_matrix = np.full((m, m), rho)
                np.fill_diagonal(corr_matrix, 1.0)
            case _:
                raise ValueError(f"Unsupported corr_type: '{corr_type}'. Use 'AR1', 'CS', or None.")

        # --- Section 3: Calculate Final Covariance Matrix ---
        if sd_vec is None:
            sd_vec = np.ones(m)

        D = np.diag(sd_vec)
        cov_matrix = D @ corr_matrix @ D  # C = D * R * D
        return cov_matrix

    @staticmethod
    def _calculate_mu_from_power(power, alpha = 0.025):
        """[Internal Helper] Calculates the non-centrality parameter (mu) from power."""
        z_crit = norm.ppf(1 - alpha)
        z_power = norm.ppf(power)
        mu = z_crit + z_power
        return mu

    @staticmethod
    def generate_p_values(
            batch_size: int,
            cov_matrix: np.ndarray = None,
            sim_alpha: float = 0.025,
            marginal_power: list | np.ndarray = None,
            power_range: float = None,
            marginal_power_lower: list | np.ndarray = None,
            marginal_power_upper: list | np.ndarray = None
    ) -> torch.Tensor:
        """
        Generates a batch of p-values with flexible power specification.

        Args:
            batch_size (int): The number of p-value vectors to generate.
            cov_matrix (np.ndarray, optional): The covariance matrix for the test statistics. Defaults to identity.
            sim_alpha (float, optional): The one-sided Type I error rate used for simulation. Defaults to 0.025.
            marginal_power (list | np.ndarray, optional): Target marginal powers for each hypothesis.
            power_range (float, optional): Symmetric range (+/-) around the target power.
            marginal_power_lower (list | np.ndarray, optional): Explicit lower bounds for power.
            marginal_power_upper (list | np.ndarray, optional): Explicit upper bounds for power.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, m) containing the generated p-values.
        """
        # --- Section 1: Determine Power Bounds ---
        lower_bound, upper_bound = None, None
        if marginal_power is not None and power_range is not None:
            target_power = np.array(marginal_power)
            lower_bound = np.maximum(0.0, target_power - power_range)
            upper_bound = np.minimum(1.0, target_power + power_range)
        elif marginal_power_lower is not None and marginal_power_upper is not None:
            lower_bound = np.array(marginal_power_lower)
            upper_bound = np.array(marginal_power_upper)
        else:
            raise ValueError(
                "You must specify power. Provide either (`marginal_power` and `power_range`) or (`marginal_power_lower` and `marginal_power_upper`).")

        # --- Section 2: Generate Random Powers and Corresponding Means (mu) ---
        m = len(lower_bound)
        random_powers = np.random.uniform(low = lower_bound, high = upper_bound, size = m)
        effective_powers = np.maximum(random_powers, sim_alpha + 1e-9)  # For numerical stability
        calculated_mus = np.array([gMCP._calculate_mu_from_power(p, sim_alpha) for p in effective_powers])
        calculated_mus[np.array(random_powers) <= sim_alpha] = 0  # Correctly simulate the null hypothesis

        # --- Section 3: Simulate Z-scores from Multivariate Normal Distribution ---
        if cov_matrix is None:
            cov_matrix = np.identity(m)
        simulated_z_scores = np.random.multivariate_normal(
            mean = calculated_mus, cov = cov_matrix, size = batch_size
        )

        # --- Section 4: Convert Z-scores to P-values ---
        p_values_np = norm.sf(simulated_z_scores)  # Survival function is 1 - CDF
        return torch.from_numpy(p_values_np)

    @staticmethod
    def graphTest(
            alpha_weight: torch.Tensor,
            T: torch.Tensor,
            p_values: torch.Tensor,
            alpha0: float
    ) -> torch.Tensor:
        """
        Implements the sequential graphical procedure.

        Args:
            alpha_weight (torch.Tensor): A vector of weights summing to 1. Shape: (m,).
            T (torch.Tensor): The transition matrix. Shape: (m, m).
            p_values (torch.Tensor): The vector of p-values. Shape: (m,).
            alpha0 (float): The total family-wise error rate.

        Returns:
            torch.Tensor: A binary decision vector. Shape: (m,).
                Example: [1., 0., 1., 0.] means H1 and H3 were rejected.
        """
        # --- Section 1: Initialization ---
        m = alpha_weight.shape[0]
        device = alpha_weight.device
        adj_alpha = alpha_weight.clone() * alpha0  # Initial alpha allocation
        decisions = torch.zeros(m, dtype = torch.float32, device = device)
        active_mask = torch.ones(m, dtype = torch.bool, device = device)

        # --- Section 2: Iterative Testing Loop ---
        while active_mask.any():
            active_indices = torch.where(active_mask)[0]
            current_adj_alpha = adj_alpha[active_mask]

            # If all remaining alphas are negligible, no more rejections are possible.
            if torch.sum(current_adj_alpha) < 1e-12:
                break

            # --- Find the next hypothesis to test ---
            ratios = p_values[active_mask] / current_adj_alpha
            ratios[torch.isinf(ratios)] = float('inf')  # Handle cases where alpha is 0
            ratios[torch.isnan(ratios)] = float('inf')

            if torch.all(torch.isinf(ratios)):
                break  # No testable hypotheses left

            min_ratio_idx_local = torch.argmin(ratios)
            j = active_indices[min_ratio_idx_local].item()  # Map local index to global index

            # --- Test the hypothesis and update graph ---
            if p_values[j] <= adj_alpha[j]:
                decisions[j] = 1.0  # Mark as rejected
                alpha_of_j = adj_alpha[j]

                # Propagate the rejected hypothesis's alpha to other active nodes
                for l_idx, l in enumerate(active_indices):
                    if l == j: continue  # Don't propagate to self
                    adj_alpha[l] += alpha_of_j * T[j, l]

            # Deactivate the current hypothesis for the next iteration, regardless of outcome.
            active_mask[j] = False

        return decisions
