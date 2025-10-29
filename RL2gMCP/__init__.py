# RL2gMCP/__init__.py
# Author: Xin-Wei Huang
# Updated: 2025-07-16

"""
This file makes the RL2gMCP directory a Python package and exposes
the main classes and functions for easy importing.
"""

# Import all key components to make them accessible at the package level
from .utils import Hyperparameters
from .gMCP import gMCP
from .action_processor import process_alpha_weight, process_T_matrix
from .agent import Agent
from .training import train_agent
from .evaluation import evaluate_policy, get_final_graph
from .translateR import plot_final_graph_in_R, plot_final_graph_in_py

# You can define what `from RL2gMCP import *` will import, which is good practice
__all__ = [
    "Hyperparameters",
    "gMCP",
    "process_alpha_weight",
    "process_T_matrix",
    "Agent",
    "train_agent",
    "evaluate_policy",
    "get_final_graph",
    "plot_final_graph_in_R",
    "plot_final_graph_in_py"
]
