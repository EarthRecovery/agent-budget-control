#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path.home()
OUT_DIR = ROOT / "figure/agent-budget-control/warehouse-gpt5.2instant"
ROLLOUT_PATH = ROOT / "database/origin/warehouse-origin-gpt5.2-instant-128-main/combined_gpt5.2-chat-latest_128seeds.json"
ESTIMATION_PATH = ROOT / "database/estimation/warehouse-OpenAI-5.2-Instant_OpenAI-5.2-Instant-128-main/warehouse-OpenAI-5.2-Instant_OpenAI-5.2-Instant-128-main.json"

COLOR_BLUE = "#1f77b4"
COLOR_ORANGE = "#ff7f0e"
COLOR_GREEN = "#2ca02c"
COLOR_RED = "#d62728"
TITLE_PREFIX = "Warehouse | GPT-5.2 Instant"


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 320,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "font.size": 10,
    }
)


def load_json(path: Path):
    return json.loads(path.read_text())


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def bool_to_num(value):
    if value is None:
        return np.nan
    return 1.0 if bool(value) else 0.0


def make_progress_bins(progress: pd.Series, bin_labels: list[str], bin_edges: np.ndarray) -> pd.Categorical:
    clipped = progress.clip(lower=0.0, upper=np.nextafter(1.0, 0.0))
    bins = pd.cut(
        clipped,
        bins=bin_edges,
        labels=bin_labels,
        right=False,
        include_lowest=True,
        ordered=True,
    )
    return pd.Categorical(bins, categories=bin_labels, ordered=True)


def extract_cached_tokens(raw_usage: dict) -> float:
    prompt_details = raw_usage.get("prompt_tokens_details") or {}
    cached_tokens = prompt_details.get("cached_tokens")
    if cached_tokens is None:
        cached_tokens = raw_usage.get("cache_read_input_tokens")
    return float(cached_tokens) if cached_tokens is not None else np.nan


def extract_raw_input_tokens(raw_usage: dict) -> float:
    prompt_tokens = raw_usage.get("prompt_tokens")
    if prompt_tokens is None:
        prompt_tokens = raw_usage.get("input_tokens")
    return float(prompt_tokens) if prompt_tokens is not None else np.nan


def extract_turn_total_tokens(turn: dict) -> float:
    api_total_tokens = turn.get("api_total_tokens")
    if api_total_tokens is not None:
        return float(api_total_tokens)

    total_tokens = 0.0
    found_any = False
    for api in turn.get("api_interactions") or []:
        api_tokens = api.get("total_tokens")
        if api_tokens is None:
            api_tokens = (api.get("usage") or {}).get("total_tokens")
        if api_tokens is not None:
            total_tokens += float(api_tokens)
            found_any = True
    return total_tokens if found_any else np.nan


def extract_turn_input_tokens(turn: dict) -> float:
    api_input_tokens = turn.get("api_input_tokens")
    if api_input_tokens is not None:
        return float(api_input_tokens)

    total_tokens = 0.0
    found_any = False
    for api in turn.get("api_interactions") or []:
        api_tokens = api.get("input_tokens")
        if api_tokens is None:
            api_tokens = (api.get("usage") or {}).get("input_tokens")
        if api_tokens is not None:
            total_tokens += float(api_tokens)
            found_any = True
    return total_tokens if found_any else np.nan


def extract_turn_output_tokens(turn: dict) -> float:
    api_output_tokens = turn.get("api_output_tokens")
    if api_output_tokens is not None:
        return float(api_output_tokens)

    total_tokens = 0.0
    found_any = False
    for api in turn.get("api_interactions") or []:
        api_tokens = api.get("output_tokens")
        if api_tokens is None:
            api_tokens = (api.get("usage") or {}).get("output_tokens")
        if api_tokens is not None:
            total_tokens += float(api_tokens)
            found_any = True
    return total_tokens if found_any else np.nan


def resolve_rollout_success(rollout: dict) -> bool:
    if rollout.get("success") is not None:
        return bool(rollout.get("success"))
    turns = rollout.get("turns") or []
    if turns and turns[-1].get("success") is not None:
        return bool(turns[-1].get("success"))
    return False


def build_rollout_df(rollouts: list[dict]) -> pd.DataFrame:
    rows = []
    for rollout_index, rollout in enumerate(rollouts):
        turns = rollout.get("turns") or []
        input_tokens_sum = 0.0
        cached_tokens_sum = 0.0
        for turn in turns:
            for api in turn.get("api_interactions") or []:
                usage = api.get("usage") or {}
                raw = usage.get("raw") or {}
                input_tokens = extract_raw_input_tokens(raw)
                if not np.isfinite(input_tokens):
                    input_tokens = usage.get("input_tokens")
                if input_tokens is not None and not pd.isna(input_tokens):
                    input_tokens_sum += float(input_tokens)
                cached_tokens = extract_cached_tokens(raw)
                if np.isfinite(cached_tokens):
                    cached_tokens_sum += float(cached_tokens)

        rows.append(
            {
                "rollout_index": rollout_index,
                "total_turns": int(rollout.get("total_turns", len(turns))),
                "success": resolve_rollout_success(rollout),
                "input_tokens_sum": input_tokens_sum,
                "cached_tokens_sum": cached_tokens_sum,
            }
        )
    return pd.DataFrame(rows)


def build_rollout_turn_df(rollouts: list[dict], bin_labels: list[str], bin_edges: np.ndarray) -> pd.DataFrame:
    rows = []
    for rollout_index, rollout in enumerate(rollouts):
        turns = rollout.get("turns") or []
        total_turns = int(rollout.get("total_turns", len(turns)) or len(turns) or 1)
        success = resolve_rollout_success(rollout)
        previous_total_tokens = 0.0
        for fallback_turn_idx, turn in enumerate(turns, start=1):
            turn_idx = int(turn.get("turn_idx", fallback_turn_idx))
            turn_total_tokens = extract_turn_total_tokens(turn)
            turn_input_tokens = extract_turn_input_tokens(turn)
            turn_output_tokens = extract_turn_output_tokens(turn)
            if np.isfinite(turn_input_tokens):
                turn_input_excluding_history = max(0.0, turn_input_tokens - previous_total_tokens)
            else:
                turn_input_excluding_history = np.nan
            turn_incremental_tokens = (
                turn_input_excluding_history + turn_output_tokens
                if np.isfinite(turn_input_excluding_history) and np.isfinite(turn_output_tokens)
                else np.nan
            )
            rows.append(
                {
                    "rollout_index": rollout_index,
                    "turn_idx": turn_idx,
                    "total_turns": total_turns,
                    "rollout_success": success,
                    "relative_progress": turn_idx / max(total_turns, 1),
                    "turn_incremental_tokens": turn_incremental_tokens,
                }
            )
            if np.isfinite(turn_total_tokens):
                previous_total_tokens = turn_total_tokens

    df = pd.DataFrame(rows)
    df["progress_bin"] = make_progress_bins(df["relative_progress"], bin_labels, bin_edges)
    return df


def build_rollout_finance_df(rollouts: list[dict]) -> pd.DataFrame:
    rows = []
    for rollout_index, rollout in enumerate(rollouts):
        turns = rollout.get("turns") or []
        total_turns = int(rollout.get("total_turns", len(turns)) or len(turns) or 1)
        for fallback_turn_idx, turn in enumerate(turns, start=1):
            turn_idx = int(turn.get("turn_idx", fallback_turn_idx))
            financials = turn.get("financials") or {}
            rows.append(
                {
                    "rollout_index": rollout_index,
                    "turn_idx": turn_idx,
                    "total_turns": total_turns,
                    "relative_progress": turn_idx / max(total_turns, 1),
                    "cumulative_warehouse_item_weeks": financials.get("cumulative_inventory_weeks", np.nan),
                    "cumulative_cost_usd": financials.get("cumulative_cost", np.nan),
                }
            )
    return pd.DataFrame(rows)


def build_estimation_df(
    estimation_payload: dict,
    rollout_df: pd.DataFrame,
    bin_labels: list[str],
    bin_edges: np.ndarray,
) -> pd.DataFrame:
    total_turn_map = rollout_df.set_index("rollout_index")["total_turns"].to_dict()
    rollout_success_map = rollout_df.set_index("rollout_index")["success"].to_dict()
    rows = []
    for result in estimation_payload["results"]:
        ground_truth = result["ground_truth"]
        prediction = result["prediction"]
        metrics = result["metrics"]
        usage = result["api_result"].get("usage") or {}
        raw_usage = usage.get("raw") or {}
        rollout_index = int(result["rollout_index"])
        turn_idx = int(result["turn_idx"])
        total_turns = int(total_turn_map[rollout_index])
        intervals = prediction.get("intervals") or {}
        pred_can_finish = prediction.get("can_finish")
        rows.append(
            {
                "sample_id": result["sample_id"],
                "rollout_index": rollout_index,
                "turn_idx": turn_idx,
                "total_turns": total_turns,
                "relative_progress": turn_idx / max(total_turns, 1),
                "rollout_success": bool(ground_truth.get("rollout_success", rollout_success_map[rollout_index])),
                "actual_can_finish": bool(ground_truth["actual_can_finish"]),
                "pred_can_finish": pred_can_finish,
                "pred_can_finish_num": bool_to_num(pred_can_finish),
                "pred_is_impossible": bool(prediction.get("is_impossible")),
                "pred_is_impossible_num": bool_to_num(prediction.get("is_impossible")),
                "pred_has_intervals_num": 1.0 if intervals else 0.0,
                "time_low": (intervals.get("time_weeks") or [np.nan, np.nan])[0],
                "time_high": (intervals.get("time_weeks") or [np.nan, np.nan])[1],
                "warehouse_low": (intervals.get("warehouse_item_weeks") or [np.nan, np.nan])[0],
                "warehouse_high": (intervals.get("warehouse_item_weeks") or [np.nan, np.nan])[1],
                "cost_low": (intervals.get("cumulative_cost_usd") or [np.nan, np.nan])[0],
                "cost_high": (intervals.get("cumulative_cost_usd") or [np.nan, np.nan])[1],
                "actual_time": ground_truth.get("actual_remaining_time_weeks"),
                "actual_warehouse": ground_truth.get("actual_remaining_warehouse_item_weeks"),
                "actual_cost": ground_truth.get("actual_remaining_cost_usd"),
                "can_finish_correct_num": bool_to_num(metrics.get("can_finish_correct")),
                "time_hit_num": bool_to_num(metrics.get("time_interval_contains_actual")),
                "warehouse_hit_num": bool_to_num(metrics.get("warehouse_interval_contains_actual")),
                "cost_hit_num": bool_to_num(metrics.get("cost_interval_contains_actual")),
                "all_hit_num": bool_to_num(metrics.get("all_intervals_cover_actual")),
                "time_hit_strict": 1.0 if metrics.get("time_interval_contains_actual") is True else 0.0,
                "warehouse_hit_strict": 1.0 if metrics.get("warehouse_interval_contains_actual") is True else 0.0,
                "cost_hit_strict": 1.0 if metrics.get("cost_interval_contains_actual") is True else 0.0,
                "all_hit_strict": 1.0 if metrics.get("all_intervals_cover_actual") is True else 0.0,
                "time_width": metrics.get("time_interval_width_weeks", np.nan),
                "warehouse_width": metrics.get("warehouse_interval_width_item_weeks", np.nan),
                "cost_width": metrics.get("cost_interval_width_usd", np.nan),
                "reward": float(metrics.get("reward", 0.0) or 0.0),
                "est_input_tokens": usage.get("input_tokens", np.nan),
                "est_output_tokens": usage.get("output_tokens", np.nan),
                "est_total_tokens": usage.get("total_tokens", np.nan),
                "est_cached_tokens": extract_cached_tokens(raw_usage),
            }
        )

    df = pd.DataFrame(rows)
    df["progress_bin"] = make_progress_bins(df["relative_progress"], bin_labels, bin_edges)
    return df


def by_bin_mean(df: pd.DataFrame, value_col: str, output_col: str, bin_labels: list[str]) -> pd.DataFrame:
    series = pd.to_numeric(df[value_col], errors="coerce")
    mean_series = series.groupby(df["progress_bin"], observed=False).mean()
    count_series = df.groupby("progress_bin", observed=False).size()
    return pd.DataFrame(
        {
            "progress_bin": bin_labels,
            output_col: [mean_series.get(label, np.nan) for label in bin_labels],
            "count": [int(count_series.get(label, 0)) for label in bin_labels],
        }
    )


def annotate_count_bars(ax: plt.Axes, xs, ys, labels) -> None:
    for x, y, label in zip(xs, ys, labels):
        if label:
            ax.text(x, y, label, ha="center", va="bottom", fontsize=8)


def classify_range_direction(low, high, actual):
    if pd.isna(low) or pd.isna(high) or pd.isna(actual):
        return None
    if float(low) <= float(actual) <= float(high):
        return "Hit"
    if float(high) < float(actual):
        return "Overly optimistic"
    if float(low) > float(actual):
        return "Overly conservative"
    return None


def plot_figure1(est_df: pd.DataFrame) -> None:
    first_turn = est_df[est_df["turn_idx"] == 1].copy()
    metric_df = pd.DataFrame(
        [
            {"metric": "Feasibility accuracy", "value": first_turn["can_finish_correct_num"].mean()},
            {"metric": "Interval prediction rate", "value": first_turn["pred_has_intervals_num"].mean()},
            {"metric": "Impossible prediction rate", "value": first_turn["pred_is_impossible_num"].mean()},
            {"metric": "Mean reward", "value": first_turn["reward"].mean()},
        ]
    )
    confusion = pd.crosstab(
        first_turn["actual_can_finish"].map({True: "Actual finish", False: "Actual impossible"}),
        first_turn["pred_can_finish"].map({True: "Predict finish", False: "Predict impossible"}),
    ).reindex(
        index=["Actual finish", "Actual impossible"],
        columns=["Predict finish", "Predict impossible"],
        fill_value=0,
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    axes[0].bar(metric_df["metric"], metric_df["value"], color=[COLOR_BLUE, COLOR_GREEN, COLOR_ORANGE, COLOR_RED], width=0.65)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Rate")
    axes[0].set_title(f"{TITLE_PREFIX} First-Turn Metrics")
    annotate_count_bars(axes[0], range(len(metric_df)), metric_df["value"].to_numpy() + 0.03, [f"{v:.2f}" for v in metric_df["value"]])
    axes[0].tick_params(axis="x", rotation=18)

    im = axes[1].imshow(confusion.values, cmap="Blues")
    axes[1].set_xticks(range(confusion.shape[1]), confusion.columns, rotation=18)
    axes[1].set_yticks(range(confusion.shape[0]), confusion.index)
    axes[1].set_title(f"{TITLE_PREFIX} First-Turn Confusion")
    for row_idx in range(confusion.shape[0]):
        for col_idx in range(confusion.shape[1]):
            axes[1].text(col_idx, row_idx, str(confusion.iloc[row_idx, col_idx]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    save(fig, "figure1_first_turn_accuracy_confusion")


def plot_figure2(est_df: pd.DataFrame, bin_labels: list[str]) -> None:
    plot_df = est_df[(est_df["rollout_success"]) & (est_df["pred_has_intervals_num"] > 0)].copy()
    time_df = by_bin_mean(plot_df, "time_hit_strict", "rate", bin_labels)
    warehouse_df = by_bin_mean(plot_df, "warehouse_hit_strict", "rate", bin_labels)
    cost_df = by_bin_mean(plot_df, "cost_hit_strict", "rate", bin_labels)
    all_df = by_bin_mean(plot_df, "all_hit_strict", "rate", bin_labels)
    fig, ax1 = plt.subplots(figsize=(10.8, 5.2))
    ax2 = ax1.twinx()
    xs = np.arange(len(bin_labels))
    ax2.bar(xs, all_df["count"], color="#d9d9d9", alpha=0.45, width=0.8, label="Samples")
    ax2.set_ylabel("Samples")
    ax2.set_ylim(0, max(all_df["count"].max() * 1.35, 1))
    ax1.plot(xs, time_df["rate"], color=COLOR_BLUE, marker="o", linewidth=2.0, label="Time hit")
    ax1.plot(xs, warehouse_df["rate"], color=COLOR_GREEN, marker="o", linewidth=2.0, label="Warehouse hit")
    ax1.plot(xs, cost_df["rate"], color=COLOR_ORANGE, marker="o", linewidth=2.0, label="Cost hit")
    ax1.plot(xs, all_df["rate"], color=COLOR_RED, marker="o", linewidth=2.2, label="All-interval hit")
    ax1.set_xticks(xs, bin_labels, rotation=35, ha="right")
    ax1.set_xlabel("Relative position in rollout")
    ax1.set_ylabel("Coverage rate")
    ax1.set_ylim(0, 1.05)
    ax1.set_title(f"{TITLE_PREFIX} Success-Rollout Interval Coverage by Relative Position (Interval Predictions Only)")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    save(fig, "figure2_interval_coverage_by_relative_position")


def plot_figure3(est_df: pd.DataFrame, bin_labels: list[str]) -> None:
    overall_df = by_bin_mean(est_df, "reward", "reward", bin_labels)
    success_df = by_bin_mean(est_df[est_df["actual_can_finish"]], "reward", "reward", bin_labels)
    fail_df = by_bin_mean(est_df[~est_df["actual_can_finish"]], "reward", "reward", bin_labels)
    fig, ax1 = plt.subplots(figsize=(10.8, 5.2))
    ax2 = ax1.twinx()
    xs = np.arange(len(bin_labels))
    ax2.bar(xs, overall_df["count"], color="#d9d9d9", alpha=0.45, width=0.8, label="Samples")
    ax2.set_ylabel("Samples")
    ax2.set_ylim(0, max(overall_df["count"].max() * 1.35, 1))
    ax1.plot(xs, overall_df["reward"], color=COLOR_BLUE, marker="o", linewidth=2.3, label="All rollouts")
    ax1.plot(xs, success_df["reward"], color=COLOR_GREEN, marker="o", linewidth=1.9, label="Actual finish")
    if fail_df["count"].sum() > 0:
        ax1.plot(xs, fail_df["reward"], color=COLOR_ORANGE, marker="o", linewidth=1.9, label="Actual impossible")
    else:
        ax1.plot([], [], color=COLOR_ORANGE, marker="o", linewidth=1.9, label="Actual impossible (no samples)")
        ax1.text(
            0.98,
            0.06,
            "No actual-impossible samples in this dataset",
            transform=ax1.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color=COLOR_ORANGE,
        )
    ax1.set_xticks(xs, bin_labels, rotation=35, ha="right")
    ax1.set_xlabel("Relative position in rollout")
    ax1.set_ylabel("Reward")
    ax1.set_ylim(0, 1.05)
    ax1.set_title(f"{TITLE_PREFIX} Reward Curve")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    save(fig, "figure3_reward_curve_by_relative_position")


def plot_figure4(est_df: pd.DataFrame, bin_labels: list[str]) -> None:
    dimensions = [
        ("time", "Time"),
        ("warehouse", "Warehouse"),
        ("cost", "Cost"),
    ]
    colors = {
        "Hit": COLOR_GREEN,
        "Overly optimistic": COLOR_ORANGE,
        "Overly conservative": "#ffbb78",
    }
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.0), sharey=True)
    count_order = ["Hit", "Overly optimistic", "Overly conservative"]
    xs = np.arange(len(bin_labels))

    for ax, (prefix, label) in zip(axes, dimensions):
        plot_df = est_df[(est_df["rollout_success"]) & (est_df["pred_has_intervals_num"] > 0)].copy()
        plot_df["range_class"] = plot_df.apply(
            lambda row: classify_range_direction(
                row[f"{prefix}_low"],
                row[f"{prefix}_high"],
                row[f"actual_{prefix}"],
            ),
            axis=1,
        )
        plot_df = plot_df[plot_df["range_class"].notna()].copy()
        counts = (
            plot_df.groupby(["progress_bin", "range_class"], observed=False)
            .size()
            .unstack(fill_value=0)
            .reindex(bin_labels)
            .fillna(0)
        )
        for column in count_order:
            if column not in counts.columns:
                counts[column] = 0
        counts = counts[count_order]
        totals = counts.sum(axis=1)
        rates = counts.div(totals.replace(0, np.nan), axis=0)
        bottom = np.zeros(len(bin_labels))
        for column in count_order:
            values = rates[column].fillna(0.0).to_numpy()
            ax.bar(xs, values, bottom=bottom, color=colors[column], width=0.78, label=column)
            bottom += values
        ax.set_xticks(xs, bin_labels, rotation=35, ha="right")
        ax.set_xlabel("Relative position")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{label} interval")
        annotate_count_bars(ax, xs, np.minimum(bottom + 0.025, 1.02), [f"n={int(v)}" if v else "" for v in totals])

    axes[0].set_ylabel("Rate within interval predictions")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(f"{TITLE_PREFIX} Hit Rate and Miss Direction", y=1.08, fontsize=14, fontweight="bold")
    save(fig, "figure4_hit_optimism_pessimism_by_dimension")


def plot_figure5(est_df: pd.DataFrame, bin_labels: list[str]) -> None:
    dimensions = [
        ("time_width", "Time width (weeks)"),
        ("warehouse_width", "Warehouse width (item-weeks)"),
        ("cost_width", "Cost width (USD)"),
    ]
    width_df = est_df[est_df["pred_has_intervals_num"] > 0].copy()
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.2))
    xs = np.arange(len(bin_labels))

    for ax, (column, title) in zip(axes, dimensions):
        summary = (
            width_df.groupby("progress_bin", observed=False)[column]
            .agg(
                median="median",
                q25=lambda values: values.quantile(0.25),
                q75=lambda values: values.quantile(0.75),
                count="size",
            )
            .reindex(bin_labels)
            .reset_index()
        )
        ax2 = ax.twinx()
        ax2.bar(xs, summary["count"].fillna(0), color="#d9d9d9", alpha=0.35, width=0.8, label="Samples")
        ax2.set_ylim(0, max(summary["count"].fillna(0).max() * 1.35, 1))
        ax2.set_ylabel("Samples")

        median = summary["median"].to_numpy(dtype=float)
        q25 = summary["q25"].to_numpy(dtype=float)
        q75 = summary["q75"].to_numpy(dtype=float)
        finite_q75 = q75[np.isfinite(q75)]
        y_max = max(float(finite_q75.max()) * 1.15, 1.0) if finite_q75.size else 1.0
        ax.fill_between(xs, q25, q75, color="#9ecae1", alpha=0.4, label="IQR")
        ax.plot(xs, median, color=COLOR_BLUE, marker="o", linewidth=2.2, label="Median width")
        ax.set_xticks(xs, bin_labels, rotation=35, ha="right")
        ax.set_xlabel("Relative position")
        ax.set_ylim(0, y_max)
        ax.set_ylabel(title)
        ax.set_title(title)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(f"{TITLE_PREFIX} Interval Width Change", y=1.08, fontsize=14, fontweight="bold")
    save(fig, "figure5_range_width_by_relative_position")


def plot_figure6(rollout_df: pd.DataFrame, est_df: pd.DataFrame) -> None:
    categories = ["Rollout", "Estimation"]
    total_inputs = [float(rollout_df["input_tokens_sum"].sum()), float(est_df["est_input_tokens"].dropna().sum())]
    cached_inputs = [float(rollout_df["cached_tokens_sum"].sum()), float(est_df["est_cached_tokens"].dropna().sum())]
    uncached_inputs = [total - cached for total, cached in zip(total_inputs, cached_inputs)]
    cached_rates = [cached / total if total else 0.0 for total, cached in zip(total_inputs, cached_inputs)]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9))
    axes[0].bar(categories, uncached_inputs, color="#3182bd", width=0.6, label="Uncached input tokens")
    axes[0].bar(categories, cached_inputs, bottom=uncached_inputs, color="#9ecae1", width=0.6, label="Cached input tokens")
    axes[0].set_ylabel("Input tokens")
    axes[0].set_title(f"{TITLE_PREFIX} Cached vs Total Input Tokens")
    axes[0].legend(loc="upper right")
    max_total = max(total_inputs) if total_inputs else 0.0
    for idx, (cached, total) in enumerate(zip(cached_inputs, total_inputs)):
        axes[0].text(
            idx,
            total + max_total * 0.02 if max_total else total,
            f"{cached/1000:.1f}k cached\n{total/1000:.1f}k total",
            ha="center",
            fontsize=9,
        )
    axes[1].bar(categories, cached_rates, color=["#4c78a8", "#f58518"], width=0.55)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_ylabel("Cached share of input tokens")
    axes[1].set_title(f"{TITLE_PREFIX} Cached Input Share")
    for idx, rate in enumerate(cached_rates):
        axes[1].text(idx, rate + 0.03, f"{rate:.1%}", ha="center")
    save(fig, "figure6_cached_tokens_rollout_vs_estimation")


def plot_figure7(turn_df: pd.DataFrame, bin_labels: list[str]) -> None:
    token_df = by_bin_mean(turn_df, "turn_incremental_tokens", "avg_tokens", bin_labels)
    fig, ax1 = plt.subplots(figsize=(10.8, 5.2))
    ax2 = ax1.twinx()
    xs = np.arange(len(bin_labels))
    ax2.bar(xs, token_df["count"], color="#d9d9d9", alpha=0.45, width=0.8, label="Turn samples")
    ax2.set_ylabel("Turn samples")
    ax2.set_ylim(0, max(token_df["count"].max() * 1.35, 1))
    ax1.plot(xs, token_df["avg_tokens"], color=COLOR_BLUE, marker="o", linewidth=2.3, label="Average tokens")
    ax1.set_xticks(xs, bin_labels, rotation=35, ha="right")
    ax1.set_xlabel("Relative position in rollout")
    ax1.set_ylabel("Average input+output tokens per turn (excluding history)")
    ax1.set_ylim(bottom=0)
    ax1.set_title(f"{TITLE_PREFIX} Average Incremental Tokens per Turn")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    save(fig, "figure7_average_tokens_used_in_rollout_turns")


def plot_figure8(est_df: pd.DataFrame, bin_labels: list[str]) -> None:
    impossible_df = by_bin_mean(est_df, "pred_is_impossible_num", "rate", bin_labels)
    fig, ax1 = plt.subplots(figsize=(10.8, 5.2))
    ax2 = ax1.twinx()
    xs = np.arange(len(bin_labels))
    ax2.bar(xs, impossible_df["count"], color="#d9d9d9", alpha=0.45, width=0.8, label="Samples")
    ax2.set_ylabel("Samples")
    ax2.set_ylim(0, max(impossible_df["count"].max() * 1.35, 1))
    ax1.plot(xs, impossible_df["rate"], color=COLOR_RED, marker="o", linewidth=2.3, label="Impossible rate")
    ax1.set_xticks(xs, bin_labels, rotation=35, ha="right")
    ax1.set_xlabel("Relative position in rollout")
    ax1.set_ylabel("Prediction rate")
    ax1.set_ylim(0, 1.05)
    ax1.set_title(f"{TITLE_PREFIX} Impossible Prediction Rate")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    save(fig, "figure8_impossible_rate_by_relative_position")


def plot_figure10(est_df: pd.DataFrame, bin_labels: list[str]) -> None:
    overall = by_bin_mean(est_df, "can_finish_correct_num", "accuracy", bin_labels)
    success = by_bin_mean(est_df[est_df["actual_can_finish"]], "can_finish_correct_num", "accuracy", bin_labels)
    failure = by_bin_mean(est_df[~est_df["actual_can_finish"]], "can_finish_correct_num", "accuracy", bin_labels)
    fig, ax1 = plt.subplots(figsize=(10.8, 5.2))
    ax2 = ax1.twinx()
    xs = np.arange(len(bin_labels))
    ax2.bar(xs, overall["count"], color="#d9d9d9", alpha=0.45, width=0.8, label="Samples")
    ax2.set_ylabel("Samples")
    ax2.set_ylim(0, max(overall["count"].max() * 1.35, 1))
    ax1.plot(xs, overall["accuracy"], color=COLOR_BLUE, marker="o", linewidth=2.3, label="Overall accuracy")
    ax1.plot(xs, success["accuracy"], color=COLOR_GREEN, marker="o", linewidth=1.9, label="Actual finish")
    ax1.plot(xs, failure["accuracy"], color=COLOR_ORANGE, marker="o", linewidth=1.9, label="Actual impossible")
    ax1.set_xticks(xs, bin_labels, rotation=35, ha="right")
    ax1.set_xlabel("Relative position in rollout")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.05)
    ax1.set_title(f"{TITLE_PREFIX} All-Turn Estimation Accuracy")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    save(fig, "figure10_all_turn_accuracy_by_relative_position")


def plot_figure9(finance_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.0), sharex=True)
    metrics = [
        ("cumulative_warehouse_item_weeks", "Cumulative warehouse_item_weeks", COLOR_GREEN),
        ("cumulative_cost_usd", "Cumulative cost (USD)", COLOR_ORANGE),
    ]

    for ax, (metric, title, color) in zip(axes, metrics):
        for idx, (rollout_index, group) in enumerate(finance_df.groupby("rollout_index", sort=True)):
            group = group.sort_values("turn_idx")
            ax.plot(
                group["turn_idx"],
                group[metric],
                color="#94a3b8",
                alpha=0.45,
                linewidth=1.2,
                label="Individual rollout" if idx == 0 else None,
            )
        mean_df = finance_df.groupby("turn_idx", sort=True)[metric].mean().reset_index()
        ax.plot(
            mean_df["turn_idx"],
            mean_df[metric],
            color=color,
            marker="o",
            linewidth=2.6,
            label="Mean",
        )
        ax.set_xlabel("Turn index")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(sorted(finance_df["turn_idx"].dropna().unique()))
        ax.legend(loc="upper left")

    fig.suptitle(f"{TITLE_PREFIX} Rollout Financial Trajectories", y=1.03, fontsize=14, fontweight="bold")
    save(fig, "figure9_rollout_warehouse_cost_by_turn")


def write_summary(rollout_df: pd.DataFrame, est_df: pd.DataFrame) -> None:
    first_turn = est_df[est_df["turn_idx"] == 1]
    lines = [
        "# Warehouse GPT-5.2 Instant Figure Summary",
        "",
        f"- Rollouts: {len(rollout_df)}",
        f"- Successful rollouts: {int(rollout_df['success'].sum())}",
        f"- Failed rollouts: {int((~rollout_df['success']).sum())}",
        f"- Average rollout turns: {rollout_df['total_turns'].mean():.2f}",
        f"- Estimation samples: {len(est_df)}",
        f"- Interval predictions: {int((est_df['pred_has_intervals_num'] > 0).sum())}",
        f"- Impossible predictions: {int(est_df['pred_is_impossible'].sum())}",
        f"- Impossible prediction rate: {est_df['pred_is_impossible_num'].mean():.3f}",
        f"- First-turn feasibility accuracy: {first_turn['can_finish_correct_num'].mean():.3f}",
        f"- Overall feasibility accuracy: {est_df['can_finish_correct_num'].mean():.3f}",
        f"- Mean reward: {est_df['reward'].mean():.3f}",
        f"- Time coverage rate (strict): {est_df['time_hit_strict'].mean():.3f}",
        f"- Warehouse coverage rate (strict): {est_df['warehouse_hit_strict'].mean():.3f}",
        f"- Cost coverage rate (strict): {est_df['cost_hit_strict'].mean():.3f}",
        f"- All-interval coverage rate (strict): {est_df['all_hit_strict'].mean():.3f}",
        "",
        f"- Rollout source: `{ROLLOUT_PATH}`",
        f"- Estimation source: `{ESTIMATION_PATH}`",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rollouts = load_json(ROLLOUT_PATH)
    estimation_payload = load_json(ESTIMATION_PATH)

    progress_bins = 10
    bin_edges = np.linspace(0.0, 1.0, progress_bins + 1)
    bin_labels = [f"{idx / progress_bins:.1f}-{(idx + 1) / progress_bins:.1f}" for idx in range(progress_bins)]

    rollout_df = build_rollout_df(rollouts)
    turn_df = build_rollout_turn_df(rollouts, bin_labels, bin_edges)
    finance_df = build_rollout_finance_df(rollouts)
    est_df = build_estimation_df(estimation_payload, rollout_df, bin_labels, bin_edges)

    plot_figure1(est_df)
    plot_figure2(est_df, bin_labels)
    plot_figure3(est_df, bin_labels)
    plot_figure4(est_df, bin_labels)
    plot_figure5(est_df, bin_labels)
    plot_figure6(rollout_df, est_df)
    plot_figure7(turn_df, bin_labels)
    plot_figure8(est_df, bin_labels)
    plot_figure10(est_df, bin_labels)
    plot_figure9(finance_df)
    write_summary(rollout_df, est_df)


if __name__ == "__main__":
    main()
