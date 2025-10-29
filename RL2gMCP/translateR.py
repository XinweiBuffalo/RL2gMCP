# RL2gMCP/translateR.py
# Author: Xin-Wei Huang
# Updated: 2025-07-16

"""
This file contains functions to translate the Python-trained graph policy
into executable R scripts and to visualize the graph directly in a Python
environment (like a Jupyter Notebook).

Functions:
- get_final_graph_in_R: Generates the R code snippet for alpha and T variables.
- plot_final_graph_in_R: Generates a full, runnable R script for plotting.
- plot_final_graph_in_py: Executes R code in the background to display the
                          graph directly in Python.
"""

import torch
import numpy as np
import os
import tempfile
import warnings

from .agent import Agent
from .evaluation import get_final_graph
from .utils import Hyperparameters

# --- Helper to check for rpy2 and configure it ---
_rpy2_installed = False
try:
    import rpy2.robjects as robjects
    from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
    import logging

    # Suppress R's console output (e.g., package loading messages)
    rpy2_logger.setLevel(logging.ERROR)
    from IPython.display import Image, display, Markdown

    _rpy2_installed = True
except ImportError:
    pass


def get_final_graph_in_R(
        agent: Agent,
        config: Hyperparameters
) -> str:
    """
    Generates an R script snippet for the final alpha and T matrix.
    """
    # This function call's inputs are the trained agent and any constraints from config.
    # It returns the final, deterministic graph for translation.
    device = config.DEVICE
    alpha_weight_fix_tensor = torch.tensor(config.ALPHA_WEIGHT_FIX, dtype = config.DTYPE,
                                           device = device) if config.ALPHA_WEIGHT_FIX is not None else None
    T_fix_tensor = torch.tensor(config.T_FIX, dtype = config.DTYPE,
                                device = device) if config.T_FIX is not None else None
    final_alpha_weight_tensor, final_T_tensor = get_final_graph(agent, alpha_weight_fix_tensor, T_fix_tensor)

    # The R function hGraph expects the final alpha vector, not the weights.
    final_alpha_tensor = final_alpha_weight_tensor * config.ALPHA_0

    alpha_np = final_alpha_tensor.numpy()
    T_np = final_T_tensor.numpy()
    m = T_np.shape[0]

    # Format alpha vector for R's c() function
    alpha_str_values = ", ".join(f"{x:.6f}" for x in alpha_np)
    r_alpha_code = f"alpha_weights_vec <- c({alpha_str_values})"

    # Format T matrix for R's matrix() function, with line breaks for readability
    T_str_values = ",\n  ".join([", ".join(map(lambda x: f"{x:.6f}", T_np[i, :])) for i in range(m)])
    r_T_matrix_code = (
        f"T_mat <- matrix(\n"
        f"  c({T_str_values}),\n"
        f"  nrow = {m}, byrow = TRUE)"
    )
    return f"{r_alpha_code}\n\n{r_T_matrix_code}"


def plot_final_graph_in_R(
        agent: Agent,
        config: Hyperparameters,
        hypo_name: list | None = None,
        output_r_script_path: str = None
) -> str:
    """
    Generates a runnable R script to plot the final graph using gMCPLite.
    """
    r_code_variables = get_final_graph_in_R(agent, config)
    m = len(config.REWARD_WEIGHTS)

    # Conditionally generate the R code for hypothesis names
    if hypo_name and len(hypo_name) == m:
        r_hypo_name_code = 'hypo_name <- c(' + ', '.join(f'"{name}"' for name in hypo_name) + ')'
    else:
        r_hypo_name_code = 'hypo_name <- paste0("H", c(1:m))'

    # Construct the full R script using the gMCPLite template
    r_code_plotting = f"""
library(tidyverse)
library(gMCPLite)

{r_code_variables}

m <- length(alpha_weights_vec)
{r_hypo_name_code}

T_mat <- T_mat %>% `rownames<-`(hypo_name) %>% `colnames<-`(hypo_name)

print(
  hGraph(
    nHypotheses = m,
    alphaHypotheses = alpha_weights_vec,
    m = T_mat
  )
)
"""
    if output_r_script_path:
        try:
            with open(output_r_script_path, 'w') as f:
                f.write(r_code_plotting)
            print(f"\nSuccessfully saved R plotting script to: {output_r_script_path}")
        except Exception as e:
            print(f"Error saving R script: {e}")
    return r_code_plotting


def plot_final_graph_in_py(
        agent: Agent,
        config: Hyperparameters,
        hypo_name: list | None = None
):
    """
    Displays the final graph directly in Python by executing R code.
    """
    if not _rpy2_installed:
        print("Required libraries not installed. Please run 'pip install rpy2'.")
        return

    print("Generating R code and executing in the background...")

    r_code_variables = get_final_graph_in_R(agent, config)
    m = len(config.REWARD_WEIGHTS)

    if hypo_name and len(hypo_name) == m:
        r_hypo_name_code = 'hypo_name <- c(' + ', '.join(f'"{name}"' for name in hypo_name) + ')'
    else:
        r_hypo_name_code = 'hypo_name <- paste0("H", c(1:m))'

    # Use a temporary file to store the plot image
    with tempfile.NamedTemporaryFile(suffix = ".png", delete = False) as tmpfile:
        temp_filename = tmpfile.name

    # Construct the R script to save the plot to the temporary file
    r_script = f"""
    suppressPackageStartupMessages(library(tidyverse))
    suppressPackageStartupMessages(library(gMCPLite))

    {r_code_variables}

    m <- length(alpha_weights_vec)
    {r_hypo_name_code}

    T_mat <- T_mat %>% `rownames<-`(hypo_name) %>% `colnames<-`(hypo_name)

    png(filename="{temp_filename.replace(os.sep, '/')}", width=8, height=6, units="in", res=150)

    print(
      hGraph(
        nHypotheses = m,
        alphaHypotheses = alpha_weights_vec,
        m = T_mat
      )
    )

    dev.off()
    """

    # Suppress rpy2 warnings and execute the script
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category = UserWarning, module = "rpy2")
        try:
            robjects.r(r_script)
            print("Plot generated successfully. Displaying result:")
            display(Image(filename = temp_filename))
        except Exception as e:
            display(Markdown(f"**An error occurred while running the R code:**\n\n"
                             f"Please ensure R and required packages are installed.\n\n"
                             f"```\n{e}\n```"))
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
