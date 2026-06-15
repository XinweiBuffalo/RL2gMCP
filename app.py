# app.py

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from RL2gMCP import (
    RLOptimizationConfig,
    run_monte_carlo_evaluation,
    run_rl_optimization,
)


st.set_page_config(
    page_title="RL2gMCP",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _default_endpoint_table(m: int) -> pd.DataFrame:
    powers = [0.95, 0.90, 0.85, 0.65, 0.60]
    weights = [20.0, 6.0, 2.0, 1.0, 1.0]
    return pd.DataFrame(
        {
            "Hypothesis": [f"H{i + 1}" for i in range(m)],
            "Marginal power": [powers[i] if i < len(powers) else 0.80 for i in range(m)],
            "Reward weight": [weights[i] if i < len(weights) else 1.0 for i in range(m)],
            "Primary": [i == 0 for i in range(m)],
            "Fixed alpha weight": [np.nan] * m,
        }
    )


def _default_t_fix_table(m: int) -> pd.DataFrame:
    return pd.DataFrame(np.full((m, m), np.nan), columns=[f"H{i + 1}" for i in range(m)], index=[f"H{i + 1}" for i in range(m)])


def _sync_tables(m: int) -> None:
    if "endpoint_table" not in st.session_state or len(st.session_state.endpoint_table) != m:
        st.session_state.endpoint_table = _default_endpoint_table(m)
    if "t_fix_table" not in st.session_state or st.session_state.t_fix_table.shape != (m, m):
        st.session_state.t_fix_table = _default_t_fix_table(m)


def _nan_to_none_vector(values: list[float]) -> list[float | None]:
    return [None if pd.isna(value) else float(value) for value in values]


def _nan_to_none_matrix(df: pd.DataFrame) -> list[list[float | None]]:
    matrix: list[list[float | None]] = []
    for row in df.to_numpy():
        matrix.append([None if pd.isna(value) else float(value) for value in row])
    return matrix


def _parse_hidden_layers(value: str) -> list[int]:
    layers = [item.strip() for item in value.split(",") if item.strip()]
    parsed = [int(item) for item in layers]
    if not parsed:
        raise ValueError("At least one hidden layer is required.")
    if any(layer <= 0 for layer in parsed):
        raise ValueError("Hidden layer sizes must be positive.")
    return parsed


def _build_config(
    endpoint_table: pd.DataFrame,
    t_fix_table: pd.DataFrame,
    *,
    power_range: float,
    alpha0: float,
    corr_type: str,
    corr_rho: float | None,
    primary_rule: str,
    reward_type: str,
    psr_threshold: float,
    psr_penalty: float,
    w_psr: float,
    w_sap: float,
    lexi_m: float,
    episodes: int,
    batch_size: int,
    learning_rate: float,
    baseline_decay: float,
    entropy_coeff: float,
    random_seed: int | None,
    device: str,
    hidden_layers_text: str,
    activation_fn: str,
    use_layer_norm: bool,
    top_k: int | None,
    dropout_rate: float | None,
    optimizer_type: str,
    weight_decay: float,
) -> RLOptimizationConfig:
    hidden_layers = _parse_hidden_layers(hidden_layers_text)
    endpoint_table = endpoint_table.copy()
    endpoint_table["Hypothesis"] = [f"H{i + 1}" for i in range(len(endpoint_table))]

    selected_corr_type = None if corr_type == "Independent" else corr_type
    selected_device = None if device == "Auto" else device.lower()

    return RLOptimizationConfig(
        marginal_power=endpoint_table["Marginal power"].astype(float).tolist(),
        power_range=power_range,
        reward_weights=endpoint_table["Reward weight"].astype(float).tolist(),
        primary_endpoints=endpoint_table["Primary"].astype(bool).astype(int).tolist(),
        primary_rule=primary_rule,
        reward_type=reward_type,
        psr_threshold=psr_threshold,
        psr_penalty=psr_penalty,
        w_psr=w_psr,
        w_sap=w_sap,
        lexi_m=lexi_m,
        corr_type=selected_corr_type,
        corr_rho=corr_rho if selected_corr_type is not None else None,
        alpha0=alpha0,
        episodes=episodes,
        batch_size=batch_size,
        learning_rate=learning_rate,
        baseline_decay=baseline_decay,
        entropy_coeff=entropy_coeff,
        random_seed=random_seed,
        device=selected_device,
        agent_hidden_layers=hidden_layers,
        agent_activation_fn=activation_fn,
        agent_use_layer_norm=use_layer_norm,
        agent_top_k=top_k,
        agent_dropout_rate=dropout_rate,
        optimizer_type=optimizer_type,
        weight_decay=weight_decay,
        alpha_weight_fix=_nan_to_none_vector(endpoint_table["Fixed alpha weight"].tolist()),
        t_fix=_nan_to_none_matrix(t_fix_table),
    )


def _format_graph_code(graph, language: str) -> str:
    alpha_values = ", ".join(f"{value:.8f}" for value in graph.alpha_weight)
    matrix_rows = [
        ", ".join(f"{value:.8f}" for value in row)
        for row in graph.transition_matrix
    ]

    if language == "R":
        matrix_values = ",\n  ".join(matrix_rows)
        return (
            f"alpha_weight <- c({alpha_values})\n"
            f"alpha <- alpha_weight * {graph.alpha0:.8f}\n\n"
            "transition_matrix <- matrix(\n"
            f"  c({matrix_values}),\n"
            f"  nrow = {len(graph.alpha_weight)}, byrow = TRUE\n"
            ")"
        )

    matrix_values = ",\n    ".join(f"[{row}]" for row in matrix_rows)
    return (
        "import numpy as np\n\n"
        f"alpha_weight = np.array([{alpha_values}])\n"
        f"alpha = alpha_weight * {graph.alpha0:.8f}\n\n"
        "transition_matrix = np.array([\n"
        f"    {matrix_values}\n"
        "])"
    )


st.title("RL2gMCP")
st.caption("Xin-Wei Huang & Meizi Liu")
run_status_placeholder = st.empty()
run_progress_placeholder = st.empty()

with st.sidebar:
    st.subheader("Testing")
    m = st.number_input("Number of hypotheses", min_value=2, max_value=12, value=5, step=1)
    _sync_tables(int(m))
    alpha0 = st.number_input("Family-wise alpha", min_value=0.001, max_value=0.25, value=0.05, step=0.005, format="%.3f")
    power_range = st.number_input("Power range", min_value=0.0, max_value=0.5, value=0.05, step=0.01, format="%.3f")
    corr_type = st.selectbox("Correlation structure", ["Independent", "AR1", "CS"], index=1)
    corr_rho = None
    if corr_type != "Independent":
        min_rho = 0.0 if corr_type == "CS" else -0.95
        corr_rho = st.number_input(
            "Correlation rho",
            min_value=min_rho,
            max_value=0.95,
            value=0.60,
            step=0.05,
            format="%.2f",
        )
    primary_rule = st.selectbox("Primary rule", ["PRIMARY", "CO_PRIMARY", "DUAL_PRIMARY"], index=0)

    st.subheader("Reward")
    reward_choice = st.selectbox("Reward type", ["PEP", "WAP"], index=0)
    reward_type = "lexicographic" if reward_choice == "PEP" else "avg_power"
    psr_threshold = st.number_input(
        "Primary power threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.90,
        step=0.01,
        format="%.2f",
        disabled=reward_type == "avg_power",
        help="Minimum desired primary power in the PEP objective.",
    )
    psr_penalty = 10.0
    w_psr = 10.0
    w_sap = 1.0
    lexi_m = 100.0
    if reward_type == "lexicographic":
        lexi_m = st.number_input(
            "Dominance constant (M)",
            min_value=0.0,
            value=100.0,
            step=10.0,
            help="Makes improvements in primary power dominate improvements in secondary power below the threshold.",
        )
    else:
        st.caption("WAP combines endpoint powers using the reward weights entered in Setup.")

    st.subheader("Training")
    episodes = st.number_input("Episodes", min_value=1, max_value=200000, value=1000, step=100)
    batch_size = st.number_input("Batch size", min_value=1, max_value=100000, value=128, step=32)
    random_seed_enabled = st.checkbox("Use random seed", value=True)
    random_seed = st.number_input("Random seed", min_value=0, value=10001, step=1, disabled=not random_seed_enabled)

    with st.expander("Advanced settings", expanded=False):
        st.caption("Training algorithm")
        learning_rate = st.number_input(
            "Learning rate",
            min_value=1e-8,
            max_value=1.0,
            value=1e-4,
            step=1e-4,
            format="%.8f",
        )
        baseline_decay = st.number_input(
            "Baseline decay",
            min_value=0.0,
            max_value=0.999,
            value=0.99,
            step=0.001,
            format="%.3f",
        )
        entropy_coeff = st.number_input(
            "Entropy coefficient",
            min_value=0.0,
            max_value=1.0,
            value=0.001,
            step=0.001,
            format="%.4f",
        )

        st.caption("Policy network")
        hidden_layers_text = st.text_input("Hidden layers", value="128, 64")
        activation_fn = st.selectbox("Activation function", ["relu", "tanh", "gelu", "elu", "leaky_relu"], index=0)
        use_layer_norm = st.checkbox("Layer normalization", value=True)
        top_k_enabled = st.checkbox("Sparse transition rows", value=False)
        top_k = st.number_input(
            "Top-k outgoing transitions",
            min_value=1,
            max_value=max(1, int(m) - 1),
            value=1,
            step=1,
            disabled=not top_k_enabled,
        )
        dropout_enabled = st.checkbox("Dropout", value=True)
        dropout_rate = st.number_input(
            "Dropout rate",
            min_value=0.0,
            max_value=0.9,
            value=0.2,
            step=0.05,
            format="%.2f",
            disabled=not dropout_enabled,
        )

        st.caption("Runtime")
        optimizer_type = st.selectbox("Optimizer", ["AdamW", "Adam"], index=0)
        weight_decay = st.number_input(
            "Weight decay",
            min_value=0.0,
            max_value=1.0,
            value=1e-2,
            step=1e-3,
            format="%.5f",
        )
        device = st.selectbox("Device", ["Auto", "CPU", "CUDA"], index=0)

about_tab, setup_tab, run_tab, eval_tab, graph_tab = st.tabs(
    ["Overview", "Setup", "Run", "Evaluation", "Optimized Graph"]
)

with about_tab:
    st.subheader("RL optimization for graphical multiple comparison procedures")
    st.write(
        "RL2gMCP helps construct and evaluate graphical multiple comparison procedures for clinical trial "
        "settings. The app trains a reinforcement learning policy to optimize the initial alpha allocation "
        "and transition matrix under user-specified assumptions, constraints, and reward criteria."
    )
    st.write(
        "Use the Setup tab to define hypotheses, marginal powers, reward weights, primary endpoints, and any "
        "fixed alpha or transition constraints. Then run RL optimization, inspect the learned graph, and perform "
        "a separate Monte Carlo evaluation."
    )
    st.info(
        "This app runs locally. Training and Monte Carlo evaluation use the CPU or GPU resources on the machine "
        "where the Streamlit app is launched."
    )
    st.subheader("Quick guide")
    st.markdown(
        """
**1. Define the clinical scenario**

- Choose the number of hypotheses, family-wise alpha, correlation structure, and plausible power range.
- In **Setup**, enter each hypothesis's marginal power and clinical importance weight.
- Mark the primary endpoint structure and optionally fix parts of the initial alpha allocation or transition matrix.

**2. Choose the optimization objective**

- **PEP** gives strict priority to achieving the primary power threshold, then improves secondary endpoint power.
- **WAP** maximizes power averaged across endpoints using the specified importance weights.

**3. Train the RL policy**

- At each episode, the policy samples a valid alpha-weight vector and transition matrix from Dirichlet distributions.
- The candidate graph is evaluated by simulated correlated p-values, and REINFORCE updates the proposal distribution
  toward higher-reward graphs.
- The final graph is the deterministic mean of the trained policy.

**4. Review and evaluate the result**

- **Optimized Graph** provides copy-ready R or Python code for the alpha vector and transition matrix.
- **Evaluation** performs an independent Monte Carlo assessment and reports primary success rate, secondary average
  power, weighted average power, and marginal rejection probabilities.
"""
    )
    st.subheader("Reward definitions")
    st.markdown("**Weighted average power (WAP)**")
    st.latex(r"\mathrm{WAP}=\sum_{i=1}^{m} v_i\,\Pr(H_i\ \mathrm{is\ rejected}), \qquad \sum_{i=1}^{m}v_i=1")
    st.write(
        "Here, m is the number of hypotheses and v_i is the normalized clinical importance weight for hypothesis i. "
        "The app normalizes the reward weights entered in Setup before calculating WAP."
    )

    st.markdown("**Primary endpoints prioritized power (PEP)**")
    st.latex(r"\mathrm{PEP}=M\min\left(\Pi_{\mathcal{P}},\,t_{\mathcal{P}}\right)+\Pi_{\mathcal{S}}")
    st.latex(
        r"\Pi_{\mathcal{S}}=\sum_{i\in\mathcal{S}}\tilde{v}_i\,"
        r"\Pr(H_i\ \mathrm{is\ rejected}), \qquad \sum_{i\in\mathcal{S}}\tilde{v}_i=1"
    )
    st.markdown(
        """
- **Pi_P**: primary power, defined by the selected primary rule. `PRIMARY` requires rejection of the single primary
  endpoint; `CO_PRIMARY` requires rejection of all primary endpoints; `DUAL_PRIMARY` requires rejection of at least
  one primary endpoint.
- **t_P**: minimum acceptable primary power, entered as **Primary power threshold**.
- **M**: dominance constant. A sufficiently large value ensures that improving primary power below the threshold
  takes priority over any gain in secondary power.
- **Pi_S**: weighted average rejection probability among secondary endpoints. Secondary weights are normalized over
  the secondary endpoints.
"""
    )

with setup_tab:
    st.subheader("Hypotheses and alpha constraints")
    endpoint_table = st.data_editor(
        st.session_state.endpoint_table,
        hide_index=True,
        num_rows="fixed",
        width="stretch",
        column_config={
            "Hypothesis": st.column_config.TextColumn(disabled=True),
            "Marginal power": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.01, format="%.3f"),
            "Reward weight": st.column_config.NumberColumn(min_value=0.0, step=0.1, format="%.3f"),
            "Primary": st.column_config.CheckboxColumn(),
            "Fixed alpha weight": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.01, format="%.3f"),
        },
        key="endpoint_editor",
    )
    st.session_state.endpoint_table = endpoint_table

    st.subheader("Transition matrix")
    st.caption(
        "Leave a cell blank (None) to let the RL agent learn that transition. Enter a number to fix that transition "
        "at the specified value during optimization. T[i, j] is the proportion transferred from hypothesis i to "
        "hypothesis j after rejection of i. Diagonal entries are always zero, and fixed off-diagonal values in each "
        "row must sum to no more than 1; the remaining row budget is allocated across blank cells."
    )
    t_fix_table = st.data_editor(
        st.session_state.t_fix_table,
        num_rows="fixed",
        width="stretch",
        column_config={
            f"H{i + 1}": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.01, format="%.3f")
            for i in range(int(m))
        },
        key="t_fix_editor",
    )
    st.session_state.t_fix_table = t_fix_table


with run_tab:
    run_col, status_col = st.columns([0.55, 0.45])
    with run_col:
        run_clicked = st.button("Run RL Optimization", type="primary", width="stretch")
    with status_col:
        if st.session_state.get("optimization_result") is not None:
            st.metric("Last training time", f"{st.session_state.optimization_result.elapsed_seconds:.1f}s")

    if run_clicked:
        try:
            run_status_placeholder.info("RL optimization started. Initializing the policy and simulations...")
            progress_bar = run_progress_placeholder.progress(0.0, text="Initializing...")

            config = _build_config(
                st.session_state.endpoint_table,
                st.session_state.t_fix_table,
                power_range=power_range,
                alpha0=alpha0,
                corr_type=corr_type,
                corr_rho=corr_rho,
                primary_rule=primary_rule,
                reward_type=reward_type,
                psr_threshold=psr_threshold,
                psr_penalty=psr_penalty,
                w_psr=w_psr,
                w_sap=w_sap,
                lexi_m=lexi_m,
                episodes=int(episodes),
                batch_size=int(batch_size),
                learning_rate=learning_rate,
                baseline_decay=baseline_decay,
                entropy_coeff=entropy_coeff,
                random_seed=int(random_seed) if random_seed_enabled else None,
                device=device,
                hidden_layers_text=hidden_layers_text,
                activation_fn=activation_fn,
                use_layer_norm=use_layer_norm,
                top_k=int(top_k) if top_k_enabled else None,
                dropout_rate=float(dropout_rate) if dropout_enabled else None,
                optimizer_type=optimizer_type,
                weight_decay=weight_decay,
            )

            progress_table = st.empty()
            progress_rows: list[dict] = []

            def progress_callback(update: dict) -> None:
                pct = min(update["episode"] / config.episodes, 1.0)
                progress_bar.progress(
                    pct,
                    text=(
                        f"Episode {update['episode']:,}/{config.episodes:,} | "
                        f"PSR {update['psr']:.3f} | SAP {update['sap']:.3f} | "
                        f"Reward {update['learning_reward']:.3f}"
                    ),
                )
                progress_rows.append(update)
                progress_table.dataframe(pd.DataFrame(progress_rows[-8:]), width="stretch", hide_index=True)

            result = run_rl_optimization(
                config,
                progress_callback=progress_callback,
                progress_gap=max(1, config.episodes // 100),
                return_agent=False,
            )
            st.session_state.optimization_result = result
            st.session_state.evaluation_result = None
            progress_bar.progress(1.0, text="Optimization complete.")
            run_status_placeholder.success(
                f"RL optimization completed in {result.elapsed_seconds:.1f} seconds."
            )
            st.success("Optimization complete.")
        except Exception as exc:
            run_progress_placeholder.empty()
            run_status_placeholder.error(f"Optimization failed: {exc}")
            st.error(str(exc))

    result = st.session_state.get("optimization_result")
    if result is not None:
        st.success(
            f"Optimization result is available. Training completed in {result.elapsed_seconds:.1f} seconds."
        )

with graph_tab:
    result = st.session_state.get("optimization_result")
    if result is None:
        st.info("Run optimization first.")
    else:
        language = st.selectbox("Programming language", ["R", "Python"], index=0)
        st.caption(
            "alpha_weight sums to 1. alpha is the initial local significance vector "
            f"(alpha_weight multiplied by family-wise alpha {result.graph.alpha0:.3f})."
        )
        st.code(
            _format_graph_code(result.graph, language),
            language="r" if language == "R" else "python",
            line_numbers=False,
        )

with eval_tab:
    st.subheader("Monte Carlo evaluation settings")
    eval_col1, eval_col2, eval_col3 = st.columns(3)
    with eval_col1:
        num_simulations = st.number_input(
            "Number of simulations",
            min_value=1,
            max_value=10000000,
            value=10000,
            step=1000,
        )
    with eval_col2:
        eval_batch_size = st.number_input(
            "Evaluation batch size",
            min_value=1,
            max_value=100000,
            value=2048,
            step=512,
        )
    with eval_col3:
        eval_seed = st.number_input("Evaluation seed", min_value=0, value=99999, step=1)

    result = st.session_state.get("optimization_result")
    if result is None:
        st.info("Run optimization first.")
    else:
        eval_clicked = st.button("Run Monte Carlo Evaluation", type="primary", width="stretch")
        if eval_clicked:
            try:
                with st.spinner("Evaluating..."):
                    evaluation = run_monte_carlo_evaluation(
                        result.graph,
                        result.config,
                        num_simulations=int(num_simulations),
                        eval_batch_size=int(eval_batch_size),
                        random_seed=int(eval_seed),
                    )
                st.session_state.evaluation_result = evaluation
                st.success("Evaluation complete.")
            except Exception as exc:
                st.error(str(exc))

        evaluation = st.session_state.get("evaluation_result")
        if evaluation is not None:
            metric_cols = st.columns(4)
            metric_cols[0].metric("PSR", f"{evaluation.psr:.3f}")
            metric_cols[1].metric("SAP", f"{evaluation.sap:.3f}")
            metric_cols[2].metric("Avg Power", f"{evaluation.avg_power:.3f}")
            metric_cols[3].metric("Time", f"{evaluation.elapsed_seconds:.1f}s")
            st.dataframe(evaluation.summary_table, width="stretch", hide_index=True)
