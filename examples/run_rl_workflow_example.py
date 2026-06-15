# examples/run_rl_workflow_example.py

"""
Minimal local example for the RL-only workflow API.

Run from the repository root:

    python examples/run_rl_workflow_example.py

Outputs are written to examples/output/.
"""

from pathlib import Path

from RL2gMCP import (
    RLOptimizationConfig,
    plot_graph,
    plot_training_history,
    run_monte_carlo_evaluation,
    run_rl_optimization,
)


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(exist_ok=True)

    config = RLOptimizationConfig(
        marginal_power=[0.90, 0.80, 0.70, 0.60],
        power_range=0.05,
        reward_weights=[0.60, 0.30, 0.05, 0.05],
        primary_endpoints=[1, 0, 0, 0],
        primary_rule="PRIMARY",
        reward_type="psr",
        psr_threshold=0.90,
        alpha0=0.05,
        corr_type="AR1",
        corr_rho=0.60,
        episodes=500,
        batch_size=128,
        random_seed=10001,
        # Use None for values that should be learned by the RL agent.
        alpha_weight_fix=[None, None, None, None],
        t_fix=[
            [None, None, None, None],
            [None, None, None, None],
            [None, None, None, None],
            [None, None, None, None],
        ],
    )

    def progress(update: dict) -> None:
        print(
            f"Episode {update['episode']:>4} | "
            f"PSR={update['psr']:.3f} | "
            f"SAP={update['sap']:.3f} | "
            f"Reward={update['learning_reward']:.3f}"
        )

    result = run_rl_optimization(config, progress_callback=progress, progress_gap=100)
    evaluation = run_monte_carlo_evaluation(result.graph, config, num_simulations=10000, random_seed=99999)

    dataframes = result.to_dataframes()
    dataframes["alpha"].to_csv(output_dir / "optimized_alpha.csv", index=False)
    dataframes["transition_matrix"].to_csv(output_dir / "optimized_transition_matrix.csv")
    result.training_history.to_csv(output_dir / "training_history.csv", index=False)
    evaluation.summary_table.to_csv(output_dir / "evaluation_summary.csv", index=False)

    training_fig = plot_training_history(result.training_history)
    training_fig.savefig(output_dir / "training_history.png", dpi=200)

    graph_fig = plot_graph(result.graph)
    graph_fig.savefig(output_dir / "optimized_graph.png", dpi=200)

    print("\nOptimized alpha weights:")
    print(dataframes["alpha"])
    print("\nEvaluation summary:")
    print(evaluation.summary_table)
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
