#!/usr/bin/env python3

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/u/ylin30")
ROLLOUT_PATH = ROOT / "database/origin/sokoban-origin-gpt5.2-instant-128-main/sokoban_api_eval_estimation_eval_estimation_dialogues.json"
ESTIMATION_PATH = ROOT / "database/estimation/sokoban-origin-gpt5.2-instant-128-main_gpt5.2-instant-token-estimation/sokoban-origin-gpt5.2-instant-128-main_gpt5.2-instant-token-estimation.json"
OUT_DIR = ROOT / "figure/agent-budget-control/figure-sokoban-gpt5.2instant"


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


def build_rollout_df(rollouts: list[dict]) -> pd.DataFrame:
    rows = []
    for idx, rollout in enumerate(rollouts):
        turns = rollout["turns"]
        rollout_input_tokens = 0
        rollout_cached_tokens = 0
        for turn in turns:
            for api in turn.get("api_interactions", []):
                raw = ((api.get("usage") or {}).get("raw") or {})
                prompt_tokens = raw.get("prompt_tokens")
                cached_tokens = ((raw.get("prompt_tokens_details") or {}).get("cached_tokens"))
                if prompt_tokens is not None:
                    rollout_input_tokens += prompt_tokens
                    rollout_cached_tokens += cached_tokens or 0
        rows.append(
            {
                "rollout_index": idx,
                "total_turns": rollout["total_turns"],
                "api_total_tokens": rollout["api_total_tokens"],
                "output_tokens_sum": sum((turn.get("actual_token") or 0) for turn in turns),
                "input_tokens_sum": rollout_input_tokens,
                "cached_tokens_sum": rollout_cached_tokens,
                "success": bool(turns and turns[-1].get("success")),
            }
        )
    return pd.DataFrame(rows)


def build_estimation_df(results: list[dict]) -> pd.DataFrame:
    rows = []
    for row in results:
        gt = row["ground_truth"]
        pred = row["prediction"]
        metrics = row["metrics"]
        usage = row["api_result"].get("usage") or {}
        input_tokens = usage.get("input_tokens", np.nan)
        output_tokens = usage.get("output_tokens", np.nan)
        total_tokens = usage.get("total_tokens", np.nan)
        cached_tokens = ((usage.get("raw", {}).get("prompt_tokens_details") or {}).get("cached_tokens", np.nan))
        interval = pred.get("remaining_token_interval") or [math.nan, math.nan]
        low, high = interval
        actual_remaining = gt["actual_remaining_total_tokens"]
        pred_can_finish = bool(pred["can_finish"] and not pred.get("is_impossible", False))
        total_ratio = np.nan
        output_ratio = np.nan
        if actual_remaining > 0 and not pd.isna(total_tokens):
            total_ratio = total_tokens / actual_remaining
        if actual_remaining > 0 and not pd.isna(output_tokens):
            output_ratio = output_tokens / actual_remaining

        rows.append(
            {
                "sample_id": row["sample_id"],
                "rollout_index": row["rollout_index"],
                "turn_idx": row["turn_idx"],
                "actual_can_finish": gt["actual_can_finish"],
                "rollout_success": gt["rollout_success"],
                "actual_tokens_used_so_far": gt["actual_tokens_used_so_far"],
                "actual_remaining_total_tokens": actual_remaining,
                "pred_can_finish": pred_can_finish,
                "pred_is_impossible": pred.get("is_impossible", False),
                "pred_low": low,
                "pred_high": high,
                "pred_width": metrics["remaining_token_interval_width"],
                "contains_actual": bool(metrics["remaining_token_interval_contains_actual"]),
                "can_finish_correct": bool(metrics["can_finish_correct"]),
                "est_input_tokens": input_tokens,
                "est_output_tokens": output_tokens,
                "est_total_tokens": total_tokens,
                "est_cached_tokens": cached_tokens,
                "est_total_vs_remaining_ratio": total_ratio,
                "est_output_vs_remaining_ratio": output_ratio,
                "stored_reward": metrics["reward"],
            }
        )

    df = pd.DataFrame(rows)
    df["custom_reward"] = df.apply(compute_custom_reward, axis=1)
    return df


def compute_custom_reward(row: pd.Series) -> float:
    if row["actual_can_finish"]:
        actual = row["actual_remaining_total_tokens"]
        low = row["pred_low"]
        high = row["pred_high"]
        if pd.isna(low) or pd.isna(high) or actual <= 0:
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
    axes[1].set_title("Success Rate vs. API Tokens")
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
    axes[0].set_title("First-Turn Rollout-Judgment Accuracy")
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
        .agg(
            overall_accuracy=("can_finish_correct", "mean"),
            count=("sample_id", "size"),
        )
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
    ax1.set_title("Rollout-Judgment Accuracy After Turn 1")
    ax1.legend(loc="upper right")

    save(fig, "figure3_later_turn_judgment_accuracy")


def plot_figure4(est_df: pd.DataFrame) -> None:
    range_df = get_success_range_df(est_df)
    range_df = range_df.sort_values(
        by=["actual_remaining_total_tokens", "pred_width"],
        ascending=[False, True],
    ).reset_index(drop=True)
    range_df["rank"] = np.arange(1, len(range_df) + 1)

    by_turn = (
        range_df.groupby("turn_idx")
        .agg(
            samples=("turn_idx", "size"),
            hits=("contains_actual", "sum"),
        )
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
        label="Predicted range",
    )
    hit_df = range_df[range_df["contains_actual"]]
    miss_df = range_df[~range_df["contains_actual"]]
    axes[0].scatter(
        hit_df["rank"],
        hit_df["actual_remaining_total_tokens"],
        color="#2ca02c",
        s=16,
        alpha=0.9,
        label="Actual tokens (hit)",
    )
    axes[0].scatter(
        miss_df["rank"],
        miss_df["actual_remaining_total_tokens"],
        color="#d62728",
        s=16,
        alpha=0.85,
        label="Actual tokens (miss)",
    )
    axes[0].set_xlabel("Predicted-success samples with actual success, sorted by actual output tokens")
    axes[0].set_ylabel("Remaining output tokens to completion")
    axes[0].set_title("Predicted Token Ranges vs. Actual Output Tokens")
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
        f"Actual Output Tokens Inside Predicted Range: {int(range_df['contains_actual'].sum())}/{len(range_df)}"
    )
    handles1, labels1 = axes[1].get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(handles1 + handles2, labels1 + labels2, loc="upper right")

    save(fig, "figure4_estimation_token_rate")


def plot_figure5(est_df: pd.DataFrame) -> None:
    summary = (
        est_df.groupby("turn_idx")
        .agg(
            custom_reward=("custom_reward", "mean"),
            stored_reward=("stored_reward", "mean"),
        )
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
    ax.set_title("Estimation Token Accuracy Reward")
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
    ax.set_title("Prediction Range Width by Turn")
    ax.legend(loc="upper right")

    save(fig, "figure5_2_estimation_range_width_by_turn")


def plot_figure6(est_df: pd.DataFrame) -> None:
    correct_ranges = get_success_range_df(est_df)
    correct_ranges = correct_ranges[correct_ranges["contains_actual"]].copy()

    correct_ranges = correct_ranges.sort_values(
        by=["actual_remaining_total_tokens", "pred_width"],
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
    ax.scatter(
        correct_ranges["rank"],
        correct_ranges["pred_low"],
        color="#3182bd",
        s=12,
        alpha=0.9,
        label="Predicted low",
    )
    ax.scatter(
        correct_ranges["rank"],
        correct_ranges["pred_high"],
        color="#08519c",
        s=12,
        alpha=0.9,
        label="Predicted high",
    )
    ax.scatter(
        correct_ranges["rank"],
        correct_ranges["actual_remaining_total_tokens"],
        color="#d62728",
        s=18,
        alpha=0.95,
        label="Actual output tokens",
    )

    ax.set_xlabel("Correct interval predictions, sorted by actual output tokens (high to low)")
    ax.set_ylabel("Remaining output tokens")
    ax.set_title("Correct Prediction Ranges vs. Actual Output Tokens")
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


def write_summary(rollout_df: pd.DataFrame, est_df: pd.DataFrame) -> None:
    first_turn = est_df[est_df["turn_idx"] == 1]
    later_turns = est_df[est_df["turn_idx"] >= 2]
    range_df = get_success_range_df(est_df)
    correct_ranges = range_df[range_df["contains_actual"]]
    excluded_false_positive_ranges = est_df[(~est_df["actual_can_finish"]) & est_df["pred_can_finish"]]
    summary_lines = [
        "# Sokoban GPT-5.2 Instant Figure Summary",
        "",
        f"- Rollouts: {len(rollout_df)}",
        f"- Rollout success rate: {rollout_df['success'].mean():.3f}",
        f"- Mean rollout turns: {rollout_df['total_turns'].mean():.2f}",
        f"- Mean rollout API total tokens: {rollout_df['api_total_tokens'].mean():.1f}",
        f"- First-turn judgment accuracy: {first_turn['can_finish_correct'].mean():.3f}",
        f"- Later-turn judgment accuracy: {later_turns['can_finish_correct'].mean():.3f}",
        f"- Mean custom reward: {est_df['custom_reward'].mean():.3f}",
        f"- Mean stored reward: {est_df['stored_reward'].mean():.3f}",
        f"- Figure 4 evaluated success-range samples: {len(range_df)}",
        f"- Figure 4 actual-token hits inside predicted range: {int(range_df['contains_actual'].sum())}",
        f"- Figure 4 hit rate: {range_df['contains_actual'].mean():.3f}",
        f"- Figure 4 excluded actual-fail but predicted-success ranges: {len(excluded_false_positive_ranges)}",
        f"- Correct interval predictions used in Figure 6: {len(correct_ranges)}",
        f"- Rollout cached-input share: {rollout_df['cached_tokens_sum'].sum() / rollout_df['input_tokens_sum'].sum():.3f}",
        f"- Estimation cached-input share: {est_df['est_cached_tokens'].dropna().sum() / est_df['est_input_tokens'].dropna().sum():.3f}",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(summary_lines))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rollouts = json.loads(ROLLOUT_PATH.read_text())
    estimation = json.loads(ESTIMATION_PATH.read_text())["results"]

    rollout_df = build_rollout_df(rollouts)
    est_df = build_estimation_df(estimation)

    plot_figure1(rollout_df)
    plot_figure2(est_df)
    plot_figure3(est_df)
    plot_figure4(est_df)
    plot_figure5(est_df)
    plot_figure52(est_df)
    plot_figure6(est_df)
    plot_figure7(rollout_df, est_df)
    write_summary(rollout_df, est_df)


if __name__ == "__main__":
    main()
