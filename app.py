# app.py

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
import tempfile

import fitz
import numpy as np
import pandas as pd
import streamlit as st

from RL2gMCP import (
    RLOptimizationConfig,
    run_monte_carlo_evaluation,
    run_rl_optimization,
)
from RL2gMCP.gMCP import gMCP


st.set_page_config(
    page_title="RL2gMCP",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_DIR = Path(__file__).resolve().parent
LOCAL_TECTONIC = APP_DIR / ".tools" / "tectonic" / "tectonic"


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
    table = pd.DataFrame(np.full((m, m), np.nan), columns=[f"H{i + 1}" for i in range(m)], index=[f"H{i + 1}" for i in range(m)])
    for idx in range(m):
        table.iat[idx, idx] = 0.0
    return table


def _enforce_zero_diagonal(df: pd.DataFrame) -> pd.DataFrame:
    fixed = df.copy()
    limit = min(fixed.shape[0], fixed.shape[1])
    for idx in range(limit):
        fixed.iat[idx, idx] = 0.0
    return fixed


def _sync_tables(m: int) -> None:
    if "endpoint_table" not in st.session_state or len(st.session_state.endpoint_table) != m:
        st.session_state.endpoint_table = _default_endpoint_table(m)
    if "t_fix_table" not in st.session_state or st.session_state.t_fix_table.shape != (m, m):
        st.session_state.t_fix_table = _default_t_fix_table(m)
    st.session_state.t_fix_table = _enforce_zero_diagonal(st.session_state.t_fix_table)


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
    corr_matrix: list[list[float]] | None,
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
        corr_rho=corr_rho if selected_corr_type in {"AR1", "CS"} else None,
        corr_matrix=corr_matrix if selected_corr_type == "CUSTOM" else None,
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


def _scenario_summary_table(config: RLOptimizationConfig) -> pd.DataFrame:
    m = len(config.marginal_power)
    fixed_alpha = config.alpha_weight_fix or [None] * m
    return pd.DataFrame(
        {
            "Hypothesis": [f"H{i + 1}" for i in range(m)],
            "Marginal power": config.marginal_power,
            "Reward weight": config.reward_weights,
            "Primary": [bool(value) for value in config.primary_endpoints],
            "Fixed alpha weight": [np.nan if value is None else value for value in fixed_alpha],
        }
    )

def _default_custom_corr_table(m: int) -> pd.DataFrame:
    names = [f"H{i + 1}" for i in range(m)]
    return pd.DataFrame(np.identity(m), index=names, columns=names)

def _sync_custom_corr_table(m: int) -> None:
    names = [f"H{i + 1}" for i in range(m)]
    table = st.session_state.get("custom_corr_table")
    if not isinstance(table, pd.DataFrame) or table.shape != (m, m) or list(table.columns) != names:
        st.session_state.custom_corr_table = _default_custom_corr_table(m)
        return
    table = table.copy()
    table.index = names
    table.columns = names
    for idx in range(m):
        table.iat[idx, idx] = 1.0
    st.session_state.custom_corr_table = table

def _custom_corr_matrix_values(df: pd.DataFrame) -> list[list[float]]:
    matrix = df.to_numpy(dtype=float)
    if np.isnan(matrix).any():
        raise ValueError("Custom correlation matrix entries must all be specified.")
    if not np.allclose(matrix, matrix.T, atol=1e-8):
        raise ValueError("Custom correlation matrix must be symmetric.")
    if not np.allclose(np.diag(matrix), np.ones(matrix.shape[0]), atol=1e-8):
        raise ValueError("Custom correlation matrix must have ones on the diagonal.")
    if np.any(matrix < -1.0 - 1e-8) or np.any(matrix > 1.0 + 1e-8):
        raise ValueError("Custom correlation matrix entries must be between -1 and 1.")
    min_eigenvalue = float(np.min(np.linalg.eigvalsh(matrix)))
    if min_eigenvalue < -1e-8:
        raise ValueError("Custom correlation matrix must be positive semidefinite.")
    return matrix.tolist()


def _matrix_for_signature(df: pd.DataFrame) -> list[list[float | None]]:
    rows: list[list[object | None]] = []
    for row in df.to_numpy(dtype=object):
        normalized_row: list[object | None] = []
        for value in row:
            if pd.isna(value):
                normalized_row.append(None)
            elif hasattr(value, "item"):
                normalized_row.append(value.item())
            else:
                normalized_row.append(value)
        rows.append(normalized_row)
    return rows


def _setup_signature(
    *,
    m: int,
    alpha0: float,
    power_range: float,
    corr_type: str,
    corr_rho: float | None,
    corr_matrix: list[list[float]] | None,
    reward_type: str,
    primary_rule: str,
    psr_threshold: float,
    lexi_m: float,
    episodes: int,
    batch_size: int,
    learning_rate: float,
    baseline_decay: float,
    entropy_coeff: float,
    random_seed: int | None,
    hidden_layers_text: str,
    activation_fn: str,
    use_layer_norm: bool,
    top_k: int | None,
    dropout_rate: float | None,
    optimizer_type: str,
    weight_decay: float,
    device: str,
    endpoint_table: pd.DataFrame,
    t_fix_table: pd.DataFrame,
) -> str:
    payload = {
        "m": int(m),
        "alpha0": float(alpha0),
        "power_range": float(power_range),
        "corr_type": corr_type,
        "corr_rho": None if corr_rho is None else float(corr_rho),
        "corr_matrix": corr_matrix,
        "reward_type": reward_type,
        "primary_rule": primary_rule,
        "psr_threshold": float(psr_threshold),
        "lexi_m": float(lexi_m),
        "episodes": int(episodes),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "baseline_decay": float(baseline_decay),
        "entropy_coeff": float(entropy_coeff),
        "random_seed": random_seed,
        "hidden_layers_text": hidden_layers_text,
        "activation_fn": activation_fn,
        "use_layer_norm": bool(use_layer_norm),
        "top_k": top_k,
        "dropout_rate": dropout_rate,
        "optimizer_type": optimizer_type,
        "weight_decay": float(weight_decay),
        "device": device,
        "endpoint_table": _matrix_for_signature(endpoint_table),
        "t_fix_table": _matrix_for_signature(t_fix_table),
    }
    return repr(payload)


def _default_evaluation_scenarios(config: RLOptimizationConfig) -> pd.DataFrame:
    data: dict[str, list[object]] = {"Scenario": ["Optimized scenario"]}
    for idx, value in enumerate(config.marginal_power):
        data[f"H{idx + 1}"] = [float(value)]
    return pd.DataFrame(data)


def _sync_evaluation_scenarios(config: RLOptimizationConfig) -> None:
    columns = ["Scenario"] + [f"H{i + 1}" for i in range(len(config.marginal_power))]
    signature = tuple(round(float(value), 8) for value in config.marginal_power)
    table = st.session_state.get("evaluation_scenarios")

    if (
        st.session_state.get("evaluation_scenarios_signature") != signature
        or not isinstance(table, pd.DataFrame)
        or list(table.columns) != columns
    ):
        st.session_state.evaluation_scenarios = _default_evaluation_scenarios(config)
        st.session_state.evaluation_scenarios_signature = signature
        return

    st.session_state.evaluation_scenarios.loc[0, "Scenario"] = "Optimized scenario"
    for idx, value in enumerate(config.marginal_power):
        st.session_state.evaluation_scenarios.loc[0, f"H{idx + 1}"] = float(value)


def _parse_evaluation_scenarios(df: pd.DataFrame, m: int) -> list[tuple[str, list[float]]]:
    scenarios: list[tuple[str, list[float]]] = []
    seen_names: set[str] = set()
    power_columns = [f"H{i + 1}" for i in range(m)]

    for row_idx, row in df.iterrows():
        values: list[float] = []
        if all(pd.isna(row[column]) for column in power_columns):
            continue
        for column in power_columns:
            value = row[column]
            if pd.isna(value):
                raise ValueError(f"Scenario row {row_idx + 1} is missing a marginal power value for {column}.")
            float_value = float(value)
            if not 0.0 <= float_value <= 1.0:
                raise ValueError(f"Marginal power values must be between 0 and 1. Invalid value in {column}.")
            values.append(float_value)
        raw_name = row.get("Scenario", "")
        scenario_name = str(raw_name).strip() if not pd.isna(raw_name) else ""
        if not scenario_name:
            scenario_name = f"Scenario {len(scenarios) + 1}"
        if scenario_name in seen_names:
            raise ValueError("Scenario names must be unique.")
        seen_names.add(scenario_name)
        scenarios.append((scenario_name, values))

    if not scenarios:
        raise ValueError("Enter at least one marginal power scenario for evaluation.")
    return scenarios


def _evaluation_summary_rows(scenarios: list[tuple[str, list[float]]], result, *, num_simulations: int, eval_batch_size: int, random_seed: int) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for scenario_name, marginal_power in scenarios:
        evaluation = run_monte_carlo_evaluation(
            result.graph,
            replace(result.config, marginal_power=marginal_power),
            num_simulations=num_simulations,
            eval_batch_size=eval_batch_size,
            random_seed=random_seed,
        )
        row: dict[str, float | str] = {
            "Scenario": scenario_name,
            "Primary endpoint success rate": evaluation.psr,
            "Secondary endpoint power": evaluation.sap,
            "Weighted average power": evaluation.avg_power,
            "Learning reward": float(evaluation.summary_table.loc[0, "Learning Reward"]),
            "Elapsed time (s)": evaluation.elapsed_seconds,
        }
        for idx, value in enumerate(evaluation.marginal_powers):
            row[f"H{idx + 1} rejection probability"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def _correlation_matrix_table(m: int, corr_type: str, corr_rho: float | None) -> pd.DataFrame:
    selected_corr_type = None if corr_type == "Independent" else corr_type
    matrix = gMCP.create_covariance_matrix(m=m, corr_type=selected_corr_type, rho=corr_rho)
    names = [f"H{i + 1}" for i in range(m)]
    return pd.DataFrame(matrix, index=names, columns=names)


def _custom_correlation_matrix_table(matrix: list[list[float]], m: int) -> pd.DataFrame:
    names = [f"H{i + 1}" for i in range(m)]
    return pd.DataFrame(np.array(matrix, dtype=float), index=names, columns=names)


def _find_tectonic() -> str | None:
    if LOCAL_TECTONIC.exists():
        return str(LOCAL_TECTONIC)
    return shutil.which("tectonic")


def _tikz_bend(i: int, j: int, m: int) -> tuple[str, int]:
    clockwise = (j - i) % m
    counter_clockwise = (i - j) % m
    distance = min(clockwise, counter_clockwise)
    angle = 20 if distance <= 1 else 15 if distance == 2 else 10
    direction = "left" if clockwise <= counter_clockwise else "right"
    return direction, angle


def _format_graph_tikz(graph) -> str:
    m = len(graph.alpha_weight)
    radius = max(4.2, 0.9 + 0.65 * m)
    angles = [90 - (360 / m) * i for i in range(m)]

    lines = [
        r"\documentclass[tikz,border=15pt]{standalone}",
        r"\usepackage{tikz}",
        r"\usetikzlibrary{arrows.meta, bending}",
        r"\begin{document}",
        r"\begin{tikzpicture}[",
        r"    hyp/.style={circle, draw, minimum size=1.7cm, inner sep=0pt, font=\Large, align=center},",
        r"    every edge/.style={draw, -{Stealth[length=5pt, width=4pt]}, semithick},",
        r"    lbl/.style={font=\small, fill=white, inner sep=1.5pt}",
        r"]",
        "",
        "% Nodes",
    ]

    for idx, angle in enumerate(angles):
        lines.append(
            rf"\node[hyp] (H{idx + 1}) at ({angle:.2f}:{radius:.2f}) "
            rf"{{\shortstack{{$H_{idx + 1}$ \\ ${graph.alpha_weight[idx]:.3f}$}}}};"
        )

    lines.extend(["", "% Edges"])
    for i in range(m):
        for j in range(i + 1, m):
            w_ij = float(graph.transition_matrix[i, j])
            w_ji = float(graph.transition_matrix[j, i])

            if w_ij <= 1e-8 and w_ji <= 1e-8:
                continue

            base_direction, base_angle = _tikz_bend(i, j, m)

            if w_ij > 1e-8:
                lines.append(
                    rf"\draw (H{i + 1}) edge[bend {base_direction}={base_angle}] "
                    rf"node[lbl, pos=0.5] {{$ {w_ij:.2f} $}} (H{j + 1});"
                )

            if w_ji > 1e-8:
                lines.append(
                    rf"\draw (H{j + 1}) edge[bend {base_direction}={base_angle}] "
                    rf"node[lbl, pos=0.5] {{$ {w_ji:.2f} $}} (H{i + 1});"
                )

    lines.extend([r"\end{tikzpicture}", r"\end{document}"])
    return "\n".join(lines)


@st.cache_data(show_spinner=False)
def _render_tikz_graph(tikz_source: str) -> tuple[bytes | None, bytes | None, str | None]:
    tectonic = _find_tectonic()
    if tectonic is None:
        return None, None, "Tectonic was not found. Install it or place it at .tools/tectonic/tectonic."

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        tex_path = tmp_dir / "graph.tex"
        pdf_path = tmp_dir / "graph.pdf"
        tex_path.write_text(tikz_source, encoding="utf-8")

        command = [tectonic, "--outdir", str(tmp_dir), str(tex_path)]
        result = subprocess.run(command, capture_output=True, text=True, cwd=tmp_dir)
        if result.returncode != 0 or not pdf_path.exists():
            error_text = (result.stderr or result.stdout or "Unknown Tectonic error").strip()
            return None, None, error_text

        pdf_bytes = pdf_path.read_bytes()
        document = fitz.open(pdf_path)
        page = document.load_page(0)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
        return pixmap.tobytes("png"), pdf_bytes, None


st.title("RL2gMCP")
st.caption("Xin-Wei Huang & Meizi Liu")
run_status_placeholder = st.empty()
run_progress_placeholder = st.empty()

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
- The marginal power uncertainty range means the simulator perturbs each entered marginal power within plus or minus that amount during training and evaluation.
- Mark the primary endpoint structure and optionally fix parts of the initial alpha allocation or transition matrix.

**2. Choose the optimization objective**

- **Primary endpoint prioritized power** gives strict priority to achieving the primary power threshold, then improves secondary endpoint power.
- **Weighted average power** maximizes power averaged across endpoints using the specified importance weights.

**3. Train the RL policy**

- At each episode, the policy samples a valid alpha-weight vector and transition matrix from Dirichlet distributions.
- The candidate graph is evaluated by simulated correlated p-values, and REINFORCE updates the proposal distribution
  toward higher-reward graphs.
- The final graph is the deterministic mean of the trained policy.

**4. Review and evaluate the result**

- **Optimized Graph** provides copy-ready R or Python code for the alpha vector and transition matrix.
- **Evaluation** performs an independent Monte Carlo assessment and reports primary endpoint success rate, secondary
    endpoint power, weighted average power, and marginal rejection probabilities.
"""
    )
    st.subheader("Reward definitions")
    st.markdown("**Weighted average power**")
    st.latex(r"\sum_{i=1}^{m} v_i\,\Pr(H_i\ \mathrm{is\ rejected}), \qquad \sum_{i=1}^{m}v_i=1")
    st.write(
        "Here, m is the number of hypotheses and v_i is the normalized clinical importance weight for hypothesis i. "
        "The app normalizes the reward weights entered in Setup before calculating weighted average power."
    )

    st.markdown("**Primary endpoint prioritized power**")
    st.latex(r"M\min\left(\Pi_{\mathcal{P}},\,t_{\mathcal{P}}\right)+\Pi_{\mathcal{S}}")
    st.latex(
        r"\Pi_{\mathcal{S}}=\sum_{i\in\mathcal{S}}\tilde{v}_i\,"
        r"\Pr(H_i\ \mathrm{is\ rejected}), \qquad \sum_{i\in\mathcal{S}}\tilde{v}_i=1"
    )
    st.markdown(
        """
- **Primary endpoint success rate**: defined by the selected primary rule. `PRIMARY` requires rejection of the single primary
  endpoint; `CO_PRIMARY` requires rejection of all primary endpoints; `DUAL_PRIMARY` requires rejection of at least
  one primary endpoint.
- **Minimum acceptable primary power**: entered as **Primary power threshold**.
- **M**: dominance constant. A sufficiently large value ensures that improving primary power below the threshold
  takes priority over any gain in secondary power.
- **Secondary endpoint power**: weighted average rejection probability among secondary endpoints. Secondary weights are normalized over
  the secondary endpoints.
"""
    )

with setup_tab:
    st.subheader("Scenario settings")
    setup_col1, setup_col2 = st.columns(2)
    with setup_col1:
        m = st.number_input("Number of hypotheses", min_value=2, max_value=12, value=5, step=1)
        alpha0 = st.number_input("Family-wise alpha", min_value=0.001, max_value=0.25, value=0.05, step=0.005, format="%.3f")
    with setup_col2:
        power_range = st.number_input(
            "Marginal power uncertainty range (+/-)",
            min_value=0.0,
            max_value=0.5,
            value=0.05,
            step=0.01,
            format="%.3f",
            help="For each hypothesis, training and evaluation simulate marginal power within the entered value plus or minus this range.",
        )

    _sync_tables(int(m))
    _sync_custom_corr_table(int(m))

    st.subheader("Reward settings")
    reward_choice = st.selectbox(
        "Reward type",
        ["Primary endpoint prioritized power", "Weighted average power"],
        index=0,
    )
    reward_type = "lexicographic" if reward_choice == "Primary endpoint prioritized power" else "avg_power"
    psr_threshold = 0.90
    psr_penalty = 10.0
    w_psr = 10.0
    w_sap = 1.0
    lexi_m = 100.0
    if reward_type == "lexicographic":
        psr_threshold = st.number_input(
            "Primary power threshold (t_p)",
            min_value=0.0,
            max_value=1.0,
            value=0.90,
            step=0.01,
            format="%.2f",
            help="Minimum desired primary power in the primary endpoint prioritized power objective.",
        )
        primary_rule = st.selectbox("Primary rule", ["PRIMARY", "CO_PRIMARY", "DUAL_PRIMARY"], index=0)
        lexi_m = st.number_input(
            "Dominance constant (M)",
            min_value=0.0,
            value=100.0,
            step=10.0,
            help="Makes improvements in primary power dominate improvements in secondary power below the threshold.",
        )
    else:
        primary_rule = "PRIMARY"
        st.caption("Weighted average power combines endpoint powers using the reward weights entered below.")

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

    st.subheader("Correlation structure")
    corr_col1, corr_col2 = st.columns(2)
    with corr_col1:
        corr_type = st.selectbox("Correlation structure", ["Independent", "AR1", "CS", "Custom"], index=1)
    with corr_col2:
        corr_rho = None
        if corr_type in {"AR1", "CS"}:
            min_rho = 0.0 if corr_type == "CS" else -0.95
            corr_rho = st.number_input(
                "Correlation rho",
                min_value=min_rho,
                max_value=0.95,
                value=0.60,
                step=0.05,
                format="%.2f",
            )
    custom_corr_matrix = None
    st.caption("The matrix below shows the covariance structure used to simulate correlated p-values during training and evaluation.")
    if corr_type == "Custom":
        st.caption("Enter a symmetric correlation matrix with ones on the diagonal and values between -1 and 1. The matrix must be positive semidefinite.")
        custom_corr_table = st.data_editor(
            st.session_state.custom_corr_table,
            num_rows="fixed",
            width="stretch",
            column_config={
                f"H{i + 1}": st.column_config.NumberColumn(min_value=-1.0, max_value=1.0, step=0.01, format="%.3f")
                for i in range(int(m))
            },
            key="custom_corr_editor",
        )
        custom_corr_table = custom_corr_table.copy()
        for idx in range(int(m)):
            custom_corr_table.iat[idx, idx] = 1.0
        st.session_state.custom_corr_table = custom_corr_table
        try:
            custom_corr_matrix = _custom_corr_matrix_values(custom_corr_table)
            st.dataframe(_custom_correlation_matrix_table(custom_corr_matrix, int(m)), width="stretch")
        except ValueError as exc:
            st.error(str(exc))
            st.dataframe(custom_corr_table, width="stretch")
    else:
        st.dataframe(_correlation_matrix_table(int(m), corr_type, corr_rho), width="stretch")

    st.subheader("Transition matrix")
    st.caption(
        "Diagonal entries are fixed at 0 and cannot be changed. Leave an off-diagonal cell blank (None) to let the RL agent learn that transition. Enter a number to fix that transition "
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
    st.session_state.t_fix_table = _enforce_zero_diagonal(t_fix_table)

    st.subheader("Training settings")
    training_col1, training_col2 = st.columns(2)
    with training_col1:
        episodes = st.number_input("Episodes", min_value=1, max_value=200000, value=2000, step=100)
        batch_size = st.number_input("Batch size", min_value=1, max_value=100000, value=512, step=32)
    with training_col2:
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

    current_setup_signature = _setup_signature(
        m=int(m),
        alpha0=alpha0,
        power_range=power_range,
        corr_type=corr_type,
        corr_rho=corr_rho,
        corr_matrix=custom_corr_matrix,
        reward_type=reward_type,
        primary_rule=primary_rule,
        psr_threshold=psr_threshold,
        lexi_m=lexi_m,
        episodes=int(episodes),
        batch_size=int(batch_size),
        learning_rate=learning_rate,
        baseline_decay=baseline_decay,
        entropy_coeff=entropy_coeff,
        random_seed=int(random_seed) if random_seed_enabled else None,
        hidden_layers_text=hidden_layers_text,
        activation_fn=activation_fn,
        use_layer_norm=use_layer_norm,
        top_k=int(top_k) if top_k_enabled else None,
        dropout_rate=float(dropout_rate) if dropout_enabled else None,
        optimizer_type=optimizer_type,
        weight_decay=weight_decay,
        device=device,
        endpoint_table=st.session_state.endpoint_table,
        t_fix_table=st.session_state.t_fix_table,
    )

    previous_signature = st.session_state.get("optimization_setup_signature")
    if st.session_state.get("optimization_result") is not None and previous_signature is not None and previous_signature != current_setup_signature:
        st.session_state.optimization_result = None
        st.session_state.evaluation_result = None
        st.session_state.pop("evaluation_scenario_highlight", None)
        st.info("Setup changed since the last optimization run. Previous optimization results were cleared.")


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
                corr_matrix=custom_corr_matrix,
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
                        f"Primary endpoint success rate {update['psr']:.3f} | "
                        f"Secondary endpoint power {update['sap']:.3f} | "
                        f"Learning reward {update['learning_reward']:.3f}"
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
            st.session_state.optimization_setup_signature = current_setup_signature
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
        tikz_source = _format_graph_tikz(result.graph)
        tikz_image, tikz_pdf, tikz_error = _render_tikz_graph(tikz_source)
        if tikz_image is not None:
            st.image(tikz_image, use_container_width=True)
            st.caption("Rendered with TikZ via Tectonic.")
            if tikz_pdf is not None:
                st.download_button(
                    "Download graph PDF",
                    data=tikz_pdf,
                    file_name="optimized-graph.pdf",
                    mime="application/pdf",
                )
        else:
            st.warning(f"TikZ rendering failed: {tikz_error}")
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
        config = result.config
        _sync_evaluation_scenarios(config)
        corr_label = "Independent" if config.corr_type is None else ("Custom" if config.corr_type == "CUSTOM" else config.corr_type)
        corr_detail = corr_label if config.corr_type in {None, "CUSTOM"} else f"{corr_label} (rho={config.corr_rho:.2f})"
        objective_label = "Primary endpoint prioritized power" if config.reward_type == "lexicographic" else "Weighted average power"
        primary_names = [f"H{i + 1}" for i, is_primary in enumerate(config.primary_endpoints) if is_primary]

        st.info(
            "Monte Carlo evaluation reuses the optimized scenario below by default. You can also add additional "
            "marginal power scenarios before running the evaluation."
        )
        summary_cols = st.columns(4)
        summary_cols[0].metric("Hypotheses", str(len(config.marginal_power)))
        summary_cols[1].metric("Family-wise alpha", f"{config.alpha0:.3f}")
        summary_cols[2].metric("Correlation", corr_detail)
        summary_cols[3].metric("Objective", objective_label)
        st.caption(
            f"Primary rule: {config.primary_rule} | Primary endpoints: {', '.join(primary_names) if primary_names else 'None'} "
            f"| Power range: {config.power_range:.3f}"
        )
        if config.reward_type == "lexicographic":
            st.caption(f"Primary power threshold: {config.psr_threshold:.2f} | Dominance constant (M): {config.lexi_m:.1f}")

        with st.expander("View endpoint assumptions", expanded=False):
            st.dataframe(_scenario_summary_table(config), width="stretch", hide_index=True)

        st.subheader("Marginal power scenarios")
        st.caption(
            "The first row is the optimized training scenario. Add rows to evaluate the same graph under additional "
            "marginal power assumptions."
        )
        evaluation_scenarios = st.data_editor(
            st.session_state.evaluation_scenarios,
            num_rows="dynamic",
            width="stretch",
            column_config={
                "Scenario": st.column_config.TextColumn("Scenario name"),
                **{
                    f"H{i + 1}": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.01, format="%.3f")
                    for i in range(len(config.marginal_power))
                },
            },
            key="evaluation_scenarios_editor",
        )
        st.session_state.evaluation_scenarios = evaluation_scenarios

        eval_clicked = st.button("Run Monte Carlo Evaluation", type="primary", width="stretch")
        if eval_clicked:
            try:
                with st.spinner("Evaluating..."):
                    scenarios = _parse_evaluation_scenarios(evaluation_scenarios, len(config.marginal_power))
                    evaluation = _evaluation_summary_rows(
                        scenarios,
                        result,
                        num_simulations=int(num_simulations),
                        eval_batch_size=int(eval_batch_size),
                        random_seed=int(eval_seed),
                    )
                st.session_state.evaluation_result = evaluation
                st.success("Evaluation complete.")
            except Exception as exc:
                st.error(str(exc))

        evaluation = st.session_state.get("evaluation_result")
        if evaluation is not None and not isinstance(evaluation, pd.DataFrame):
            st.session_state.evaluation_result = None
            evaluation = None
            st.info("Previous evaluation results were cleared because the evaluation table format changed. Run the evaluation again.")

        if evaluation is not None:
            selected_scenario = st.selectbox("Scenario to highlight", evaluation["Scenario"].tolist(), key="evaluation_scenario_highlight")
            selected_row = evaluation.loc[evaluation["Scenario"] == selected_scenario].iloc[0]
            metric_cols = st.columns(4)
            metric_cols[0].metric("Primary endpoint success rate", f"{selected_row['Primary endpoint success rate']:.3f}")
            metric_cols[1].metric("Secondary endpoint power", f"{selected_row['Secondary endpoint power']:.3f}")
            metric_cols[2].metric("Weighted average power", f"{selected_row['Weighted average power']:.3f}")
            metric_cols[3].metric("Elapsed time", f"{selected_row['Elapsed time (s)']:.1f}s")
            st.dataframe(evaluation, width="stretch", hide_index=True)
