#!/usr/bin/env python3

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path.home()
ROLLOUT_PATH = ROOT / "database/origin/sokoban-origin-gpt5.2-instant-128-main-new/sokoban_api_eval_estimation_eval_estimation_dialogues.json"
ESTIMATION_PATH = ROOT / "database/estimation/sokoban-origin-gpt5.2-instant-128-main-new_gpt5.2-instant-token-estimation-test/sokoban-origin-gpt5.2-instant-128-main-new_gpt5.2-instant-token-estimation-test.json"
OUT_DIR = ROOT / "figure/agent-budget-control/figure-sokoban-gpt5.2instant-new"


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "font.size": 10,
    }
)


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def _safe_bool_or_nan(value):
    if value is None:
        return np.nan
    return bool(value)


def build_rollout_df(rollouts: list[dict]) -> pd.DataFrame:
    rows = []
    for idx, rollout in enumerate(rollouts):
        turns = rollout.get("turns") or []
        rollout_input_tokens = 0
        rollout_cached_tokens = 0
        final_turn_total_tokens = turns[-1].get("api_total_tokens") if turns else np.nan
        for turn in turns:
            for api in turn.get("api_interactions", []):
                raw = ((api.get("usage") or {}).get("raw") or {})
                prompt_tokens = raw.get("prompt_tokens")
                cached_tokens = ((raw.get("prompt_tokens_details") or {}).get("cached_tokens"))
                if prompt_tokens is not None:
                    rollout_input_tokens += int(prompt_tokens)
                    rollout_cached_tokens += int(cached_tokens or 0)
        rows.append(
            {
                "rollout_index": idx,
                "total_turns": rollout.get("total_turns", len(turns)),
                "api_total_tokens": rollout.get("api_total_tokens"),
                "final_turn_total_tokens": final_turn_total_tokens,
                "input_tokens_sum": rollout_input_tokens,
                "cached_tokens_sum": rollout_cached_tokens,
                "success": bool(turns and turns[-1].get("success")),
            }
        )
    return pd.DataFrame(rows)


def build_estimation_df(results: list[dict], max_context_window_tokens: int) -> pd.DataFrame:
    rows = []
    for row in results:
        gt = row["ground_truth"]
        pred = row["prediction"]
        metrics = row["metrics"]
        usage = row["api_result"].get("usage") or {}
        interval = pred.get("remaining_token_interval") or [math.nan, math.nan]
        low, high = interval
        actual_growth = gt["actual_remaining_total_tokens"]
        pred_can_finish_raw = pred.get("can_finish")
        pred_is_impossible = bool(pred.get("is_impossible", False))
        pred_can_finish = bool(pred_can_finish_raw) and not pred_is_impossible
        total_tokens = usage.get("total_tokens", np.nan)
        input_tokens = usage.get("input_tokens", np.nan)
        output_tokens = usage.get("output_tokens", np.nan)
        cached_tokens = ((usage.get("raw", {}).get("prompt_tokens_details") or {}).get("cached_tokens", np.nan))
        footprint_before_current_turn = gt["actual_tokens_used_so_far"]
        terminal_footprint = (
            float(footprint_before_current_turn) + float(actual_growth)
            if actual_growth is not None and not pd.isna(actual_growth)
            else np.nan
        )

        total_ratio = np.nan
        if actual_growth and not pd.isna(total_tokens):
            total_ratio = float(total_tokens) / float(actual_growth)

        rows.append(
            {
                "sample_id": row["sample_id"],
                "rollout_index": row["rollout_index"],
                "turn_idx": row["turn_idx"],
                "actual_can_finish": bool(gt["actual_can_finish"]),
                "rollout_success": bool(gt["rollout_success"]),
                "footprint_before_current_turn": footprint_before_current_turn,
                "actual_growth_to_finishing_turn": actual_growth,
                "terminal_footprint": terminal_footprint,
                "budget_gap_before_turn": float(footprint_before_current_turn) - float(max_context_window_tokens),
                "remaining_budget_before_turn": float(max_context_window_tokens) - float(footprint_before_current_turn),
                "pred_can_finish": pred_can_finish,
                "pred_is_impossible": pred_is_impossible,
                "pred_low": low,
                "pred_high": high,
                "pred_width": metrics.get("remaining_token_interval_width"),
                "contains_actual": _safe_bool_or_nan(metrics.get("remaining_token_interval_contains_actual")),
                "can_finish_correct": _safe_bool_or_nan(metrics.get("can_finish_correct")),
                "est_input_tokens": input_tokens,
                "est_output_tokens": output_tokens,
                "est_total_tokens": total_tokens,
                "est_cached_tokens": cached_tokens,
                "est_total_vs_growth_ratio": total_ratio,
                "stored_reward": metrics.get("reward", np.nan),
            }
        )

    df = pd.DataFrame(rows)
    df["custom_reward"] = df.apply(compute_custom_reward, axis=1)
    return df


def compute_custom_reward(row: pd.Series) -> float:
    if row["actual_can_finish"]:
        actual = row["actual_growth_to_finishing_turn"]
        low = row["pred_low"]
        high = row["pred_high"]
        if pd.isna(low) or pd.isna(high) or actual is None or actual <= 0:
            return 0.0
        if low <= actual <= high:
            return max(0.0, 1.0 - ((high - low) / actual))
        return 0.0
    return 1.0 if row["pred_is_impossible"] else 0.0


def get_success_range_df(est_df: pd.DataFrame) -> pd.DataFrame:
    return est_df[
        est_df["actual_can_finish"]
        & est_df["pred_can_finish"]
        & est_df["pred_low"].notna()
        & est_df["pred_high"].notna()
    ].copy()


def classify_miss_direction(row: pd.Series) -> str:
    if row["contains_actual"] == True:
        return "Hit"
    if pd.isna(row["pred_low"]) or pd.isna(row["pred_high"]):
        return "No interval"
    if row["pred_high"] < row["actual_growth_to_finishing_turn"]:
        return "Optimistic miss"
    if row["pred_low"] > row["actual_growth_to_finishing_turn"]:
        return "Conservative miss"
    return "Other miss"


def assign_rollout_stage(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("turn_idx").copy()
    max_turn = int(group["turn_idx"].max())
    denom = max(max_turn - 1, 1)
    progress = (group["turn_idx"] - 1) / denom

    def _stage(value: float) -> str:
        if value < (1.0 / 3.0):
            return "early"
        if value < (2.0 / 3.0):
            return "mid"
        return "late"

    group["stage"] = progress.map(_stage)
    return group


def select_case_study_rollouts(
    est_df: pd.DataFrame,
    rollout_df: pd.DataFrame,
    max_context_window_tokens: int,
) -> dict[str, int]:
    rows = []
    rollout_token_map = rollout_df.set_index("rollout_index")["api_total_tokens"].to_dict()
    for rollout_index, group in est_df.groupby("rollout_index"):
        group = group.sort_values("turn_idx")
        pred_success_count = int(group["pred_can_finish"].sum())
        hit_count = int(((group["contains_actual"] == True) & group["pred_can_finish"]).sum())
        impossible_turns = group.loc[~group["pred_can_finish"], "turn_idx"].tolist()
        first_impossible_turn = impossible_turns[0] if impossible_turns else np.nan
        rows.append(
            {
                "rollout_index": rollout_index,
                "actual_can_finish": bool(group["actual_can_finish"].iloc[0]),
                "turns": len(group),
                "pred_success_count": pred_success_count,
                "hit_count": hit_count,
                "all_pred_success": pred_success_count == len(group),
                "first_impossible_turn": first_impossible_turn,
                "rollout_total_tokens": rollout_token_map.get(rollout_index, np.nan),
            }
        )

    rollout_df = pd.DataFrame(rows)

    success_good = (
        rollout_df[rollout_df["actual_can_finish"]]
        .sort_values(["hit_count", "all_pred_success", "turns"], ascending=[False, False, False])
        .iloc[0]["rollout_index"]
    )

    success_bad_candidates = rollout_df[
        rollout_df["actual_can_finish"]
        & rollout_df["all_pred_success"]
        & (rollout_df["hit_count"] == 0)
    ]
    if success_bad_candidates.empty:
        success_bad_candidates = rollout_df[rollout_df["actual_can_finish"]].sort_values(
            ["hit_count", "turns"], ascending=[True, False]
        )
    success_bad = success_bad_candidates.sort_values(["turns"], ascending=[False]).iloc[0]["rollout_index"]

    fail_late_candidates = rollout_df[
        (~rollout_df["actual_can_finish"])
        & rollout_df["first_impossible_turn"].notna()
        & (rollout_df["rollout_total_tokens"] > max_context_window_tokens)
    ]
    if fail_late_candidates.empty:
        fail_late_candidates = rollout_df[
            (~rollout_df["actual_can_finish"])
            & rollout_df["first_impossible_turn"].notna()
        ]
    fail_late = fail_late_candidates.sort_values(
        ["first_impossible_turn", "turns"], ascending=[False, False]
    ).iloc[0]["rollout_index"]

    fail_never_candidates = rollout_df[
        (~rollout_df["actual_can_finish"])
        & rollout_df["all_pred_success"]
        & (rollout_df["rollout_total_tokens"] > max_context_window_tokens)
    ]
    if fail_never_candidates.empty:
        fail_never_candidates = rollout_df[
            (~rollout_df["actual_can_finish"])
            & rollout_df["all_pred_success"]
        ]
    fail_never = fail_never_candidates.sort_values(["turns"], ascending=[False]).iloc[0]["rollout_index"]

    return {
        "success_good": int(success_good),
        "success_bad": int(success_bad),
        "fail_late": int(fail_late),
        "fail_never": int(fail_never),
    }


def plot_figure1(rollout_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    by_turn = (
        rollout_df.groupby("total_turns")
        .agg(success_rate=("success", "mean"), count=("success", "size"))
        .reset_index()
    )
    axes[0].bar(by_turn["total_turns"], by_turn["success_rate"], color="#1f77b4", alpha=0.85)
    axes[0].set_title("Success Rate vs. Rollout Turns")
    axes[0].set_xlabel("Total rollout turns")
    axes[0].set_ylabel("Success rate")
    axes[0].set_ylim(0, 1.05)
    for _, row in by_turn.iterrows():
        axes[0].text(row["total_turns"], row["success_rate"] + 0.03, f"n={int(row['count'])}", ha="center", fontsize=8)

    max_token = int(math.ceil(rollout_df["api_total_tokens"].max() / 3000.0) * 3000)
    bins = np.arange(0, max_token + 3000, 3000)
    token_bins = pd.cut(rollout_df["api_total_tokens"], bins=bins, right=False, include_lowest=True)
    by_token = (
        rollout_df.assign(token_bin=token_bins)
        .dropna(subset=["token_bin"])
        .groupby("token_bin", observed=False)
        .agg(success_rate=("success", "mean"), count=("success", "size"))
        .reset_index()
    )
    labels = [f"{int(interval.left/1000)}k-{int(interval.right/1000)}k" for interval in by_token["token_bin"]]
    axes[1].bar(labels, by_token["success_rate"], color="#ff7f0e", alpha=0.85)
    axes[1].set_title("Success Rate vs. Rollout API Tokens")
    axes[1].set_xlabel("Total API tokens per rollout")
    axes[1].set_ylabel("Success rate")
    axes[1].set_ylim(0, 1.05)
    axes[1].tick_params(axis="x", rotation=30)
    for x, (_, row) in enumerate(by_token.iterrows()):
        axes[1].text(x, row["success_rate"] + 0.03, f"n={int(row['count'])}", ha="center", fontsize=8)

    save(fig, "figure1_rollout_success_vs_turns_tokens")


def plot_figure2(est_df: pd.DataFrame) -> None:
    first_turn = est_df[est_df["turn_idx"] == 1].copy()

    metrics_df = pd.DataFrame(
        [
            {"group": "Overall", "accuracy": first_turn["can_finish_correct"].mean()},
            {
                "group": "Actual success",
                "accuracy": first_turn[first_turn["actual_can_finish"]]["can_finish_correct"].mean(),
            },
            {
                "group": "Actual fail",
                "accuracy": first_turn[~first_turn["actual_can_finish"]]["can_finish_correct"].mean(),
            },
        ]
    )

    confusion = pd.crosstab(
        first_turn["actual_can_finish"].map({True: "Actual success", False: "Actual fail"}),
        first_turn["pred_can_finish"].map({True: "Predict success", False: "Predict fail"}),
    ).reindex(index=["Actual success", "Actual fail"], columns=["Predict success", "Predict fail"], fill_value=0)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

    axes[0].bar(metrics_df["group"], metrics_df["accuracy"], color=["#4c78a8", "#2ca02c", "#d62728"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("First-Turn Finishing-Turn Judgment Accuracy")
    for x, row in metrics_df.iterrows():
        axes[0].text(x, row["accuracy"] + 0.03, f"{row['accuracy']:.2f}", ha="center")

    im = axes[1].imshow(confusion.values, cmap="Blues")
    axes[1].set_xticks(range(confusion.shape[1]), confusion.columns, rotation=20)
    axes[1].set_yticks(range(confusion.shape[0]), confusion.index)
    axes[1].set_title("First-Turn Confusion Matrix")
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            axes[1].text(j, i, confusion.iloc[i, j], ha="center", va="center", color="black")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    save(fig, "figure2_first_turn_judgment_accuracy")


def plot_figure3(est_df: pd.DataFrame) -> None:
    other_turns = est_df[est_df["turn_idx"] >= 2].copy()
    summary = (
        other_turns.groupby("turn_idx")
        .agg(overall_accuracy=("can_finish_correct", "mean"), count=("sample_id", "size"))
        .reset_index()
    )
    success_summary = (
        other_turns[other_turns["actual_can_finish"]]
        .groupby("turn_idx")
        .agg(success_accuracy=("can_finish_correct", "mean"))
        .reset_index()
    )
    fail_summary = (
        other_turns[~other_turns["actual_can_finish"]]
        .groupby("turn_idx")
        .agg(fail_accuracy=("can_finish_correct", "mean"))
        .reset_index()
    )

    fig, ax1 = plt.subplots(figsize=(8.8, 4.8))
    ax2 = ax1.twinx()
    ax2.bar(summary["turn_idx"], summary["count"], color="#d9d9d9", alpha=0.45, width=0.75, label="Samples")
    ax2.set_ylabel("Samples")
    ax2.set_ylim(0, summary["count"].max() * 1.35)

    ax1.plot(summary["turn_idx"], summary["overall_accuracy"], marker="o", linewidth=2.2, color="#1f77b4", label="Overall")
    ax1.plot(success_summary["turn_idx"], success_summary["success_accuracy"], marker="o", linewidth=1.8, color="#2ca02c", label="Actual success")
    ax1.plot(fail_summary["turn_idx"], fail_summary["fail_accuracy"], marker="o", linewidth=1.8, color="#d62728", label="Actual fail")
    ax1.set_xlabel("Turn index")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Finishing-Turn Judgment Accuracy After Turn 1")
    ax1.legend(loc="upper right")

    save(fig, "figure3_later_turn_judgment_accuracy")


def plot_figure4(est_df: pd.DataFrame) -> None:
    range_df = get_success_range_df(est_df)
    range_df = range_df.sort_values(
        by=["actual_growth_to_finishing_turn", "pred_width"],
        ascending=[False, True],
    ).reset_index(drop=True)
    range_df["rank"] = np.arange(1, len(range_df) + 1)

    by_turn = (
        range_df.groupby("turn_idx")
        .agg(samples=("turn_idx", "size"), hits=("contains_actual", "sum"))
        .reset_index()
    )
    by_turn["misses"] = by_turn["samples"] - by_turn["hits"]
    by_turn["hit_rate"] = by_turn["hits"] / by_turn["samples"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    axes[0].vlines(
        range_df["rank"],
        range_df["pred_low"],
        range_df["pred_high"],
        color="#9ecae1",
        alpha=0.8,
        linewidth=1.1,
        label="Predicted growth range",
    )
    hit_df = range_df[range_df["contains_actual"] == True]
    miss_df = range_df[range_df["contains_actual"] != True]
    axes[0].scatter(
        hit_df["rank"],
        hit_df["actual_growth_to_finishing_turn"],
        color="#2ca02c",
        s=16,
        alpha=0.9,
        label="Actual growth (hit)",
    )
    axes[0].scatter(
        miss_df["rank"],
        miss_df["actual_growth_to_finishing_turn"],
        color="#d62728",
        s=16,
        alpha=0.85,
        label="Actual growth (miss)",
    )
    axes[0].set_xlabel("Predicted-success samples with actual success, sorted by actual growth")
    axes[0].set_ylabel("Additional total tokens to finishing turn")
    axes[0].set_title("Predicted Growth Ranges vs. Actual Growth")
    axes[0].set_xlim(0, len(range_df) + 1)
    axes[0].legend(loc="upper right")

    ax2 = axes[1].twinx()
    axes[1].bar(by_turn["turn_idx"], by_turn["hits"], color="#2ca02c", width=0.7, label="Hits")
    axes[1].bar(by_turn["turn_idx"], by_turn["misses"], bottom=by_turn["hits"], color="#fcae91", width=0.7, label="Misses")
    ax2.plot(by_turn["turn_idx"], by_turn["hit_rate"], color="#1f77b4", marker="o", linewidth=2, label="Hit rate")
    axes[1].set_xlabel("Turn index")
    axes[1].set_ylabel("Prediction count")
    ax2.set_ylabel("Hit rate")
    ax2.set_ylim(0, 1.05)
    axes[1].set_title(
        f"Actual Growth Inside Predicted Range: {int(range_df['contains_actual'].sum())}/{len(range_df)}"
    )
    handles1, labels1 = axes[1].get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(handles1 + handles2, labels1 + labels2, loc="upper right")

    save(fig, "figure4_estimation_token_rate")


def plot_figure5(est_df: pd.DataFrame) -> None:
    summary = (
        est_df.groupby("turn_idx")
        .agg(custom_reward=("custom_reward", "mean"), stored_reward=("stored_reward", "mean"))
        .reset_index()
    )
    success_summary = (
        est_df[est_df["actual_can_finish"]]
        .groupby("turn_idx")
        .agg(success_reward=("custom_reward", "mean"))
        .reset_index()
    )
    fail_summary = (
        est_df[~est_df["actual_can_finish"]]
        .groupby("turn_idx")
        .agg(fail_reward=("custom_reward", "mean"))
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(summary["turn_idx"], summary["custom_reward"], marker="o", linewidth=2.2, color="#1f77b4", label="Custom reward")
    ax.plot(summary["turn_idx"], summary["stored_reward"], marker="s", linewidth=1.8, color="#7f7f7f", linestyle="--", label="Stored reward")
    ax.plot(success_summary["turn_idx"], success_summary["success_reward"], marker="o", linewidth=1.6, color="#2ca02c", label="Actual success only")
    ax.plot(fail_summary["turn_idx"], fail_summary["fail_reward"], marker="o", linewidth=1.6, color="#d62728", label="Actual fail only")
    ax.set_xlabel("Turn index")
    ax.set_ylabel("Mean reward")
    ax.set_ylim(0, 1.05)
    ax.set_title("Estimation Reward by Turn")
    ax.legend(loc="upper right")

    save(fig, "figure5_estimation_token_accuracy_reward")


def plot_figure52(est_df: pd.DataFrame) -> None:
    turns = sorted(est_df["turn_idx"].unique())
    groups = [est_df.loc[est_df["turn_idx"] == turn, "pred_width"].dropna().values for turn in turns]
    medians = est_df.groupby("turn_idx")["pred_width"].median().reset_index()

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.boxplot(groups, positions=turns, showfliers=False, widths=0.6, patch_artist=True, boxprops={"facecolor": "#c7e9c0"})
    ax.plot(medians["turn_idx"], medians["pred_width"], color="#238b45", marker="o", linewidth=2, label="Median width")
    ax.set_yscale("log")
    ax.set_xlabel("Turn index")
    ax.set_ylabel("Predicted interval width (log scale)")
    ax.set_title("Prediction Growth-Range Width by Turn")
    ax.legend(loc="upper right")

    save(fig, "figure5_2_estimation_range_width_by_turn")


def plot_figure6(est_df: pd.DataFrame) -> None:
    correct_ranges = get_success_range_df(est_df)
    correct_ranges = correct_ranges[correct_ranges["contains_actual"] == True].copy()
    correct_ranges = correct_ranges.sort_values(
        by=["actual_growth_to_finishing_turn", "pred_width"],
        ascending=[False, True],
    ).reset_index(drop=True)
    correct_ranges["rank"] = np.arange(1, len(correct_ranges) + 1)

    fig, ax = plt.subplots(figsize=(12, 5.2))

    ax.vlines(
        correct_ranges["rank"],
        correct_ranges["pred_low"],
        correct_ranges["pred_high"],
        color="#9ecae1",
        alpha=0.9,
        linewidth=1.3,
        label="Predicted range",
    )
    ax.scatter(correct_ranges["rank"], correct_ranges["pred_low"], color="#3182bd", s=12, alpha=0.9, label="Predicted low")
    ax.scatter(correct_ranges["rank"], correct_ranges["pred_high"], color="#08519c", s=12, alpha=0.9, label="Predicted high")
    ax.scatter(
        correct_ranges["rank"],
        correct_ranges["actual_growth_to_finishing_turn"],
        color="#d62728",
        s=18,
        alpha=0.95,
        label="Actual growth",
    )

    ax.set_xlabel("Correct interval predictions, sorted by actual growth (high to low)")
    ax.set_ylabel("Additional total tokens to finishing turn")
    ax.set_title("Correct Prediction Ranges vs. Actual Growth")
    ax.set_xlim(0, len(correct_ranges) + 1)
    ax.legend(loc="upper right", ncol=2)

    save(fig, "figure6_correct_range_vs_actual_tokens")


def plot_figure7(rollout_df: pd.DataFrame, est_df: pd.DataFrame) -> None:
    categories = ["Rollout", "Estimation"]
    total_inputs = [
        rollout_df["input_tokens_sum"].sum(),
        est_df["est_input_tokens"].dropna().sum(),
    ]
    cached_inputs = [
        rollout_df["cached_tokens_sum"].sum(),
        est_df["est_cached_tokens"].dropna().sum(),
    ]
    uncached_inputs = [total - cached for total, cached in zip(total_inputs, cached_inputs)]
    cached_rates = [
        cached / total if total else 0.0
        for total, cached in zip(total_inputs, cached_inputs)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    axes[0].bar(categories, uncached_inputs, color="#9ecae1", label="Uncached input tokens")
    axes[0].bar(categories, cached_inputs, bottom=uncached_inputs, color="#3182bd", label="Cached input tokens")
    axes[0].set_ylabel("Input tokens")
    axes[0].set_title("Cached vs. Total Input Tokens")
    axes[0].legend(loc="upper right")
    for idx, (total, cached) in enumerate(zip(total_inputs, cached_inputs)):
        axes[0].text(idx, total + max(total_inputs) * 0.02, f"{cached/1000:.1f}k cached\n{total/1000:.1f}k total", ha="center", fontsize=9)

    axes[1].bar(categories, cached_rates, color=["#4c78a8", "#f58518"], width=0.55)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_ylabel("Cached share of input tokens")
    axes[1].set_title("Cached Input Token Rate")
    for idx, rate in enumerate(cached_rates):
        axes[1].text(idx, rate + 0.03, f"{rate:.1%}", ha="center")

    save(fig, "figure7_cached_tokens_vs_total_input_tokens")


def plot_figure8(est_df: pd.DataFrame) -> None:
    summary = (
        est_df.groupby(["turn_idx", "actual_can_finish"])
        .agg(pred_success_rate=("pred_can_finish", "mean"), count=("sample_id", "size"))
        .reset_index()
    )
    total = est_df.groupby("turn_idx").agg(total_samples=("sample_id", "size")).reset_index()
    success = summary[summary["actual_can_finish"]].copy()
    fail = summary[~summary["actual_can_finish"]].copy()

    fig, ax1 = plt.subplots(figsize=(8.8, 4.8))
    ax2 = ax1.twinx()
    ax2.bar(total["turn_idx"], total["total_samples"], color="#d9d9d9", alpha=0.45, width=0.78, label="Samples")
    ax2.set_ylabel("Samples")
    ax2.set_ylim(0, total["total_samples"].max() * 1.35)

    ax1.plot(
        success["turn_idx"],
        success["pred_success_rate"],
        marker="o",
        linewidth=2.2,
        color="#2ca02c",
        label="Actual success -> predict success",
    )
    ax1.plot(
        fail["turn_idx"],
        fail["pred_success_rate"],
        marker="o",
        linewidth=2.2,
        color="#d62728",
        label="Actual fail -> predict success",
    )
    ax1.set_xlabel("Turn index")
    ax1.set_ylabel("Predicted-success rate")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Selective Failure by Turn")
    ax1.legend(loc="upper right")

    save(fig, "figure8_selective_failure_by_turn")


def plot_figure9(est_df: pd.DataFrame) -> None:
    fail_df = est_df[~est_df["actual_can_finish"]].copy()
    fail_df = fail_df[fail_df["remaining_budget_before_turn"] >= 0].copy()
    bins = [0, 500, 1000, 1500, 100_000]
    labels = ["0:500", "500:1000", "1000:1500", "1500+"]
    fail_df["remaining_budget_bucket"] = pd.cut(
        fail_df["remaining_budget_before_turn"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )
    summary = (
        fail_df.groupby("remaining_budget_bucket", observed=False)
        .agg(pred_success_rate=("pred_can_finish", "mean"), count=("sample_id", "size"))
        .reset_index()
    )

    colors = plt.cm.OrRd(np.linspace(0.35, 0.85, len(summary)))
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(summary["remaining_budget_bucket"].astype(str), summary["pred_success_rate"], color=colors, width=0.7)
    ax.set_xlabel("Remaining budget before this turn (pre-cutoff states only)")
    ax.set_ylabel("False-positive rate on actual-fail samples")
    ax.set_ylim(0, 1.05)
    ax.set_title("False Positives Before Hard Budget Cutoff")
    for idx, row in summary.iterrows():
        ax.text(idx, row["pred_success_rate"] + 0.03, f"n={int(row['count'])}", ha="center", fontsize=8)
    ax.tick_params(axis="x", rotation=20)

    save(fig, "figure9_false_positives_vs_budget_pressure")


def plot_figure10(est_df: pd.DataFrame) -> None:
    success_df = est_df[est_df["actual_can_finish"]].copy()
    success_df = (
        success_df.groupby("rollout_index", group_keys=False)
        .apply(assign_rollout_stage)
        .reset_index(drop=True)
    )
    range_df = success_df[
        success_df["pred_can_finish"]
        & success_df["pred_low"].notna()
        & success_df["pred_high"].notna()
    ].copy()

    stage_order = ["early", "mid", "late"]
    width_groups = [range_df.loc[range_df["stage"] == stage, "pred_width"].dropna().values for stage in stage_order]
    summary = (
        range_df.groupby("stage")
        .agg(
            hit_rate=("contains_actual", "mean"),
            median_width=("pred_width", "median"),
            count=("sample_id", "size"),
        )
        .reindex(stage_order)
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))

    axes[0].boxplot(
        width_groups,
        positions=np.arange(1, len(stage_order) + 1),
        showfliers=False,
        widths=0.55,
        patch_artist=True,
        boxprops={"facecolor": "#c7e9c0"},
    )
    axes[0].plot(
        np.arange(1, len(stage_order) + 1),
        summary["median_width"],
        color="#238b45",
        marker="o",
        linewidth=2,
        label="Median width",
    )
    axes[0].set_xticks(np.arange(1, len(stage_order) + 1), [s.title() for s in stage_order])
    axes[0].set_ylabel("Predicted interval width")
    axes[0].set_title("Range Width on Actual-Success Rollouts")
    axes[0].legend(loc="upper right")

    bars = axes[1].bar(summary["stage"].str.title(), summary["hit_rate"], color=["#9ecae1", "#6baed6", "#3182bd"], width=0.62)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Hit rate")
    axes[1].set_title("Hit Rate on Actual-Success Rollouts")
    for bar, (_, row) in zip(bars, summary.iterrows()):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            f"n={int(row['count'])}",
            ha="center",
            fontsize=9,
        )

    fig.suptitle("Actual-Success Rollouts by Early / Mid / Late Stage", y=1.02, fontsize=14, fontweight="bold")

    save(fig, "figure10_actual_success_stage_range_hit_rate")


def plot_figure11(est_df: pd.DataFrame) -> None:
    range_df = get_success_range_df(est_df)
    summary = (
        range_df.groupby("turn_idx")
        .agg(
            median_width=("pred_width", "median"),
            hit_rate=("contains_actual", "mean"),
            count=("sample_id", "size"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    sizes = 35 + summary["count"] * 4.0
    scatter = ax.scatter(
        summary["median_width"],
        summary["hit_rate"],
        s=sizes,
        c=summary["turn_idx"],
        cmap="viridis",
        alpha=0.9,
        edgecolors="black",
        linewidths=0.5,
    )
    for _, row in summary.iterrows():
        ax.text(row["median_width"] + 12, row["hit_rate"] + 0.012, str(int(row["turn_idx"])), fontsize=8)
    ax.set_xlabel("Median predicted interval width")
    ax.set_ylabel("Interval hit rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Coverage-Tightness Tradeoff by Turn")
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Turn index")

    save(fig, "figure11_coverage_tightness_tradeoff")


def plot_figure12(rollout_df: pd.DataFrame, est_df: pd.DataFrame, max_context_window_tokens: int) -> None:
    selected = select_case_study_rollouts(est_df, rollout_df, max_context_window_tokens)
    final_turn_context_map = rollout_df.set_index("rollout_index")["final_turn_total_tokens"].to_dict()
    panel_specs = [
        ("success_good", "Success / Well-calibrated"),
        ("success_bad", "Success / Persistent miss"),
        ("fail_late", "Fail / Warns only at the end"),
        ("fail_never", "Fail / Never warns"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8), sharey=True)
    axes = axes.flatten()

    for ax, (key, label) in zip(axes, panel_specs):
        rollout_index = selected[key]
        group = est_df[est_df["rollout_index"] == rollout_index].sort_values("turn_idx").copy()
        group["pred_low_footprint"] = group["footprint_before_current_turn"] + group["pred_low"]
        group["pred_high_footprint"] = group["footprint_before_current_turn"] + group["pred_high"]
        actual_terminal = final_turn_context_map.get(rollout_index, np.nan)
        if pd.isna(actual_terminal):
            actual_terminal = group["terminal_footprint"].dropna().iloc[0]
        predict_success = group[group["pred_can_finish"] & group["pred_low"].notna() & group["pred_high"].notna()]
        predict_impossible = group[~group["pred_can_finish"]]

        ax.axhline(actual_terminal, color="black", linewidth=1.6, label="Final-turn total context length")
        ax.axhline(
            max_context_window_tokens,
            color="#d62728",
            linestyle="--",
            linewidth=1.3,
            label="Budget",
        )
        ax.vlines(
            predict_success["turn_idx"],
            predict_success["pred_low_footprint"],
            predict_success["pred_high_footprint"],
            color="#4c78a8",
            linewidth=2.0,
            alpha=0.85,
            label="Predicted final-turn context range",
        )
        ax.scatter(
            predict_success["turn_idx"],
            np.full(len(predict_success), actual_terminal),
            color="black",
            s=18,
            zorder=3,
        )
        if not predict_impossible.empty:
            ax.scatter(
                predict_impossible["turn_idx"],
                np.full(len(predict_impossible), max_context_window_tokens),
                marker="x",
                s=44,
                linewidths=1.4,
                color="#d62728",
                label="Predict impossible",
                zorder=4,
            )

        outcome_text = "actual success" if bool(group["actual_can_finish"].iloc[0]) else "actual fail"
        ax.set_title(f"{label}\nrollout {rollout_index} ({outcome_text})")
        ax.set_xlabel("Turn index")
        ax.set_xlim(0.5, group["turn_idx"].max() + 0.5)
        ax.set_ylim(
            0,
            max(
                rollout_df["final_turn_total_tokens"].dropna().quantile(0.98),
                max_context_window_tokens * 1.15,
                actual_terminal * 1.05,
            ),
        )

    axes[0].set_ylabel("Implied final-turn total context length")
    axes[2].set_ylabel("Implied final-turn total context length")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Representative Trajectory Case Studies", y=1.06, fontsize=14, fontweight="bold")

    save(fig, "figure12_representative_trajectory_case_studies")


def plot_figure13(est_df: pd.DataFrame) -> None:
    composition = (
        est_df.groupby(["turn_idx", "actual_can_finish"])
        .agg(count=("sample_id", "size"))
        .reset_index()
    )
    composition_pivot = (
        composition.pivot(index="turn_idx", columns="actual_can_finish", values="count")
        .fillna(0)
        .rename(columns={True: "Actual success", False: "Actual fail"})
    )

    fail_pred = (
        est_df[~est_df["actual_can_finish"]]
        .groupby(["turn_idx", "pred_can_finish"])
        .agg(count=("sample_id", "size"))
        .reset_index()
    )
    fail_pred_pivot = (
        fail_pred.pivot(index="turn_idx", columns="pred_can_finish", values="count")
        .fillna(0)
        .rename(columns={True: "Predict success", False: "Predict impossible"})
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True)

    axes[0].bar(
        composition_pivot.index,
        composition_pivot.get("Actual success", pd.Series(index=composition_pivot.index, dtype=float)),
        color="#2ca02c",
        width=0.72,
        label="Actual success",
    )
    axes[0].bar(
        composition_pivot.index,
        composition_pivot.get("Actual fail", pd.Series(index=composition_pivot.index, dtype=float)),
        bottom=composition_pivot.get("Actual success", 0),
        color="#d62728",
        width=0.72,
        label="Actual fail",
    )
    axes[0].set_title("Sample Composition by Turn")
    axes[0].set_xlabel("Turn index")
    axes[0].set_ylabel("Samples")
    axes[0].legend(loc="upper right")

    axes[1].bar(
        fail_pred_pivot.index,
        fail_pred_pivot.get("Predict success", pd.Series(index=fail_pred_pivot.index, dtype=float)),
        color="#f58518",
        width=0.72,
        label="False positive: predict success",
    )
    axes[1].bar(
        fail_pred_pivot.index,
        fail_pred_pivot.get("Predict impossible", pd.Series(index=fail_pred_pivot.index, dtype=float)),
        bottom=fail_pred_pivot.get("Predict success", 0),
        color="#4c78a8",
        width=0.72,
        label="Predict impossible",
    )
    axes[1].set_title("Actual-Fail Samples by Model Judgment")
    axes[1].set_xlabel("Turn index")
    axes[1].set_ylabel("Samples")
    axes[1].legend(loc="upper right")

    save(fig, "figure13_sample_composition_by_turn")


def write_summary(rollout_df: pd.DataFrame, est_df: pd.DataFrame) -> None:
    first_turn = est_df[est_df["turn_idx"] == 1]
    later_turns = est_df[est_df["turn_idx"] >= 2]
    range_df = get_success_range_df(est_df)
    correct_ranges = range_df[range_df["contains_actual"] == True]
    excluded_false_positive_ranges = est_df[(~est_df["actual_can_finish"]) & est_df["pred_can_finish"]]
    summary_lines = [
        "# Sokoban GPT-5.2 Instant New Figure Summary",
        "",
        f"- Rollouts: {len(rollout_df)}",
        f"- Rollout success rate: {rollout_df['success'].mean():.3f}",
        f"- Mean rollout turns: {rollout_df['total_turns'].mean():.2f}",
        f"- Mean rollout API total tokens: {rollout_df['api_total_tokens'].mean():.1f}",
        f"- First-turn judgment accuracy: {first_turn['can_finish_correct'].mean():.3f}",
        f"- Later-turn judgment accuracy: {later_turns['can_finish_correct'].mean():.3f}",
        f"- Mean custom reward: {est_df['custom_reward'].mean():.3f}",
        f"- Mean stored reward: {est_df['stored_reward'].mean():.3f}",
        f"- Success-range samples: {len(range_df)}",
        f"- Actual-growth hits inside predicted range: {int(range_df['contains_actual'].sum())}",
        f"- Hit rate: {range_df['contains_actual'].mean():.3f}",
        f"- False-positive predicted-success ranges: {len(excluded_false_positive_ranges)}",
        f"- Correct interval predictions used in Figure 6: {len(correct_ranges)}",
        f"- Rollout cached-input share: {rollout_df['cached_tokens_sum'].sum() / rollout_df['input_tokens_sum'].sum():.3f}",
        f"- Estimation cached-input share: {est_df['est_cached_tokens'].dropna().sum() / est_df['est_input_tokens'].dropna().sum():.3f}",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(summary_lines))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rollouts = json.loads(ROLLOUT_PATH.read_text())
    estimation_payload = json.loads(ESTIMATION_PATH.read_text())
    estimation = estimation_payload["results"]
    max_context_window_tokens = int(estimation_payload["config"]["max_context_window_tokens"])

    rollout_df = build_rollout_df(rollouts)
    est_df = build_estimation_df(estimation, max_context_window_tokens)

    plot_figure1(rollout_df)
    plot_figure2(est_df)
    plot_figure3(est_df)
    plot_figure4(est_df)
    plot_figure5(est_df)
    plot_figure52(est_df)
    plot_figure6(est_df)
    plot_figure7(rollout_df, est_df)
    plot_figure8(est_df)
    plot_figure9(est_df)
    plot_figure10(est_df)
    plot_figure11(est_df)
    plot_figure12(rollout_df, est_df, max_context_window_tokens)
    plot_figure13(est_df)
    write_summary(rollout_df, est_df)


if __name__ == "__main__":
    main()
