#!/usr/bin/env python3

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/u/ylin30")
FIGURE_ROOT = ROOT / "figure/agent-budget-control"
OVERALL_DIR = FIGURE_ROOT / "overall"
DATABASE_ORIGIN_ROOT = ROOT / "database/origin"
DATABASE_ESTIMATION_ROOT = ROOT / "database/estimation"

COLOR_BLUE = "#1f77b4"
COLOR_ORANGE = "#ff7f0e"
COLOR_GREEN = "#2ca02c"


@dataclass(frozen=True)
class DatasetConfig:
    slug: str
    task: str
    task_display: str
    model_display: str
    rollout_path: Path
    estimation_path: Path
    progress_bins: int
    reward_mode: str

    @property
    def out_dir(self) -> Path:
        return FIGURE_ROOT / self.slug

    @property
    def title_prefix(self) -> str:
        return f"{self.task_display} | {self.model_display}"


CONFIGS = [
    DatasetConfig(
        slug="searchr1-gpt5.2instant",
        task="searchr1",
        task_display="SearchR1",
        model_display="GPT-5.2 Instant",
        rollout_path=ROOT / "database/origin/searchr1-origin-gpt5.2-instant-128-main/search_r1_api_eval_estimation_eval_estimation_dialogues.json",
        estimation_path=ROOT / "database/estimation/searchr1-origin-gpt5.2-instant-128-main_gpt5.2-instant-token-estimation-main/searchr1-origin-gpt5.2-instant-128-main_gpt5.2-instant-token-estimation-main.json",
        progress_bins=5,
        reward_mode="stored",
    ),
    DatasetConfig(
        slug="searchr1-gpt5.2instant-new-prompt",
        task="searchr1",
        task_display="SearchR1",
        model_display="GPT-5.2 Instant New Prompt",
        rollout_path=ROOT / "database/origin/searchr1-origin-gpt5.2-instant-128-main/search_r1_api_eval_estimation_eval_estimation_dialogues.json",
        estimation_path=ROOT / "database/estimation-test/searchr1-origin-gpt5.2-instant-128-main_gpt5.2-instant-token-estimation-main2/searchr1-origin-gpt5.2-instant-128-main_gpt5.2-instant-token-estimation-main2.json",
        progress_bins=5,
        reward_mode="stored",
    ),
    DatasetConfig(
        slug="searchr1-claude-opus-4.7-low-thinking",
        task="searchr1",
        task_display="SearchR1",
        model_display="Claude Opus 4.7 Low Thinking",
        rollout_path=ROOT / "database/origin/searchr1-origin-Claude-Opus-4.7-low-thinking-128-main/search_r1_api_eval_estimation_eval_estimation_dialogues.json",
        estimation_path=ROOT / "database/estimation/searchr1-origin-Claude-Opus-4.7-low-thinking-128-main_Claude-Opus-4.7-low-thinking-128-token-estimation-main/searchr1-origin-Claude-Opus-4.7-low-thinking-128-main_Claude-Opus-4.7-low-thinking-128-token-estimation-main.json",
        progress_bins=5,
        reward_mode="stored",
    ),
    DatasetConfig(
        slug="searchr1-claude-sonnet-4.6-low-thinking",
        task="searchr1",
        task_display="SearchR1",
        model_display="Claude Sonnet 4.6 Low Thinking",
        rollout_path=ROOT / "database/origin/searchr1-origin-Claude-Sonnet-4.6-low-thinking-128-main/search_r1_api_eval_estimation_eval_estimation_dialogues.json",
        estimation_path=ROOT / "database/estimation/searchr1-origin-Claude-Sonnet-4.6-low-thinking-128-main_Claude-Sonnet-4.6-low-thinking-128-token-estimation-main/searchr1-origin-Claude-Sonnet-4.6-low-thinking-128-main_Claude-Sonnet-4.6-low-thinking-128-token-estimation-main.json",
        progress_bins=5,
        reward_mode="stored",
    ),
    DatasetConfig(
        slug="sokoban-gpt5.2instant",
        task="sokoban",
        task_display="Sokoban",
        model_display="GPT-5.2 Instant",
        rollout_path=ROOT / "database/origin/sokoban-origin-gpt5.2-instant-128-main/sokoban_api_eval_estimation_eval_estimation_dialogues.json",
        estimation_path=ROOT / "database/estimation/sokoban-origin-gpt5.2-instant-128-main_gpt5.2-instant-128-token-estimation-main/sokoban-origin-gpt5.2-instant-128-main_gpt5.2-instant-128-token-estimation-main.json",
        progress_bins=10,
        reward_mode="sokoban_custom",
    ),
    DatasetConfig(
        slug="sokoban-claude-opus-4.7-low-thinking",
        task="sokoban",
        task_display="Sokoban",
        model_display="Claude Opus 4.7 Low Thinking",
        rollout_path=ROOT / "database/origin/sokoban-origin-claude-opus-4.7-low-thinking-128-main/sokoban_api_eval_estimation_eval_estimation_dialogues.json",
        estimation_path=ROOT / "database/estimation/sokoban-origin-claude-opus-4.7-low-thinking-128-main_origin-claude-opus-4.7-low-thinking-token-estimation-main/sokoban-origin-claude-opus-4.7-low-thinking-128-main_origin-claude-opus-4.7-low-thinking-token-estimation-main.json",
        progress_bins=10,
        reward_mode="sokoban_custom",
    ),
    DatasetConfig(
        slug="sokoban-claude-sonnet-4.6-low-thinking",
        task="sokoban",
        task_display="Sokoban",
        model_display="Claude Sonnet 4.6 Low Thinking",
        rollout_path=ROOT / "database/origin/sokoban-origin-claude-sonnet-4.6-low-thinking-128-main/sokoban_api_eval_estimation_eval_estimation_dialogues.json",
        estimation_path=ROOT / "database/estimation/sokoban-origin-claude-sonnet-4.6-low-thinking-128-main_origin-claude-sonnet-4.6-low-thinking-token-estimation-main/sokoban-origin-claude-sonnet-4.6-low-thinking-128-main_origin-claude-sonnet-4.6-low-thinking-token-estimation-main.json",
        progress_bins=10,
        reward_mode="sokoban_custom",
    ),
]

CONFIG_BY_SLUG = {cfg.slug: cfg for cfg in CONFIGS}


def is_overall_config(cfg: DatasetConfig) -> bool:
    try:
        cfg.rollout_path.relative_to(DATABASE_ORIGIN_ROOT)
        cfg.estimation_path.relative_to(DATABASE_ESTIMATION_ROOT)
    except ValueError:
        return False
    return True


OVERALL_CONFIGS = [cfg for cfg in CONFIGS if is_overall_config(cfg)]


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


def save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


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
    turns = rollout.get("turns") or []
    if turns and turns[-1].get("success") is not None:
        return bool(turns[-1].get("success"))
    final_state = str(rollout.get("final_state") or "")
    if "Reward: 1.00" in final_state:
        return True
    if "Boxes on target:" in final_state and "Boxes:" not in final_state:
        return True
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
                "total_turns": rollout.get("total_turns", len(turns)),
                "api_total_tokens": rollout.get("api_total_tokens", np.nan),
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
                    "turn_total_tokens": turn_total_tokens,
                    "turn_input_excluding_history": turn_input_excluding_history,
                    "turn_output_tokens": turn_output_tokens,
                    "turn_incremental_tokens": turn_incremental_tokens,
                }
            )
            previous_total_tokens = turn_total_tokens if np.isfinite(turn_total_tokens) else previous_total_tokens

    df = pd.DataFrame(rows)
    df["progress_bin"] = make_progress_bins(df["relative_progress"], bin_labels, bin_edges)
    return df


def compute_sokoban_reward(row: pd.Series) -> float:
    if bool(row["actual_can_finish"]):
        actual = row["actual_remaining_total_tokens"]
        low = row["pred_low"]
        high = row["pred_high"]
        if pd.isna(low) or pd.isna(high) or actual is None or pd.isna(actual) or float(actual) <= 0:
            return 0.0
        if float(low) <= float(actual) <= float(high):
            width = max(0.0, float(high) - float(low))
            return max(0.0, 1.0 - (width / float(actual)))
        return 0.0
    return 1.0 if bool(row["pred_is_impossible"]) else 0.0


def compute_custom_reward(row: pd.Series) -> float:
    if bool(row["actual_can_finish"]):
        actual = row["actual_remaining_total_tokens"]
        low = row["pred_low"]
        high = row["pred_high"]
        if pd.isna(low) or pd.isna(high) or actual is None or pd.isna(actual) or float(actual) <= 0:
            return 0.0
        if float(low) <= float(actual) <= float(high):
            width = max(0.0, float(high) - float(low))
            return max(0.0, 1.0 - (width / float(actual)))
        return 0.0
    return 1.0 if bool(row["pred_is_impossible"]) else 0.0


def build_estimation_df(
    estimation_payload: dict,
    bin_labels: list[str],
    bin_edges: np.ndarray,
    reward_fn: Optional[Callable[[pd.Series], float]],
) -> pd.DataFrame:
    rows = []
    for result in estimation_payload["results"]:
        ground_truth = result["ground_truth"]
        prediction = result["prediction"]
        metrics = result["metrics"]
        usage = result["api_result"].get("usage") or {}
        raw_usage = usage.get("raw") or {}
        interval = prediction.get("remaining_token_interval") or [math.nan, math.nan]
        pred_low, pred_high = interval
        pred_is_impossible = bool(prediction.get("is_impossible", False))
        pred_can_finish = bool(prediction.get("can_finish")) and not pred_is_impossible

        rows.append(
            {
                "sample_id": result["sample_id"],
                "rollout_index": result["rollout_index"],
                "turn_idx": result["turn_idx"],
                "relative_progress": float(ground_truth["relative_progress"]),
                "completed_turns": int(ground_truth["completed_turns"]),
                "total_turns": int(ground_truth["total_turns"]),
                "rollout_success": bool(ground_truth["rollout_success"]),
                "actual_can_finish": bool(ground_truth["actual_can_finish"]),
                "actual_remaining_total_tokens": ground_truth["actual_remaining_total_tokens"],
                "pred_can_finish": pred_can_finish,
                "pred_is_impossible": pred_is_impossible,
                "pred_low": pred_low,
                "pred_high": pred_high,
                "pred_width": metrics.get("remaining_token_interval_width", np.nan),
                "contains_actual": metrics.get("remaining_token_interval_contains_actual"),
                "can_finish_correct": metrics.get("can_finish_correct"),
                "stored_reward": metrics.get("reward", np.nan),
                "est_input_tokens": usage.get("input_tokens", np.nan),
                "est_output_tokens": usage.get("output_tokens", np.nan),
                "est_total_tokens": usage.get("total_tokens", np.nan),
                "est_cached_tokens": extract_cached_tokens(raw_usage),
            }
        )

    df = pd.DataFrame(rows)
    if reward_fn is None:
        df["reward"] = df["stored_reward"]
    else:
        df["reward"] = df.apply(reward_fn, axis=1)
    df["custom_reward"] = df.apply(compute_custom_reward, axis=1)
    df["progress_bin"] = make_progress_bins(df["relative_progress"], bin_labels, bin_edges)
    return df


def by_bin_mean(df: pd.DataFrame, value_col: str, output_col: str, bin_labels: list[str]) -> pd.DataFrame:
    mean_series = df.groupby("progress_bin", observed=False)[value_col].mean()
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
        if not label:
            continue
        ax.text(x, y, label, ha="center", va="bottom", fontsize=8)


def classify_range_direction(row: pd.Series) -> str:
    if row["contains_actual"] is True:
        return "Hit"
    if row["pred_high"] < row["actual_remaining_total_tokens"]:
        return "Overly optimistic"
    if row["pred_low"] > row["actual_remaining_total_tokens"]:
        return "Overly conservative"
    return "Other"


def plot_figure1(cfg: DatasetConfig, est_df: pd.DataFrame, out_dir: Path) -> None:
    first_turn = est_df[est_df["turn_idx"] == 1].copy()
    accuracy_df = pd.DataFrame(
        [
            {"group": "Overall", "accuracy": first_turn["can_finish_correct"].mean()},
            {"group": "Success rollout", "accuracy": first_turn[first_turn["rollout_success"]]["can_finish_correct"].mean()},
            {"group": "Failure rollout", "accuracy": first_turn[~first_turn["rollout_success"]]["can_finish_correct"].mean()},
        ]
    )
    confusion = pd.crosstab(
        first_turn["rollout_success"].map({True: "Actual success", False: "Actual failure"}),
        first_turn["pred_can_finish"].map({True: "Predict finish", False: "Predict fail"}),
    ).reindex(
        index=["Actual success", "Actual failure"],
        columns=["Predict finish", "Predict fail"],
        fill_value=0,
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))
    axes[0].bar(accuracy_df["group"], accuracy_df["accuracy"], color=[COLOR_BLUE, COLOR_GREEN, COLOR_ORANGE], width=0.65)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title(f"{cfg.title_prefix} First-Turn Estimation Accuracy")
    annotate_count_bars(axes[0], range(len(accuracy_df)), accuracy_df["accuracy"] + 0.03, [f"{value:.2f}" for value in accuracy_df["accuracy"]])
    axes[0].tick_params(axis="x", rotation=15)

    im = axes[1].imshow(confusion.values, cmap="Blues")
    axes[1].set_xticks(range(confusion.shape[1]), confusion.columns, rotation=20)
    axes[1].set_yticks(range(confusion.shape[0]), confusion.index)
    axes[1].set_title(f"{cfg.title_prefix} First-Turn Confusion Matrix")
    for row_idx in range(confusion.shape[0]):
        for col_idx in range(confusion.shape[1]):
            axes[1].text(col_idx, row_idx, str(confusion.iloc[row_idx, col_idx]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    save(fig, out_dir, "figure1_first_turn_accuracy_confusion")


def plot_figure2(cfg: DatasetConfig, est_df: pd.DataFrame, out_dir: Path, bin_labels: list[str]) -> None:
    overall = by_bin_mean(est_df, "can_finish_correct", "accuracy", bin_labels)
    success = by_bin_mean(est_df[est_df["rollout_success"]], "can_finish_correct", "accuracy", bin_labels)
    failure = by_bin_mean(est_df[~est_df["rollout_success"]], "can_finish_correct", "accuracy", bin_labels)
    fig, ax1 = plt.subplots(figsize=(10.8, 5.2))
    ax2 = ax1.twinx()
    xs = np.arange(len(bin_labels))
    ax2.bar(xs, overall["count"], color="#d9d9d9", alpha=0.45, width=0.8, label="Samples")
    ax2.set_ylabel("Samples")
    ax2.set_ylim(0, max(overall["count"].max() * 1.35, 1))
    ax1.plot(xs, overall["accuracy"], color=COLOR_BLUE, marker="o", linewidth=2.3, label="Overall accuracy")
    ax1.plot(xs, success["accuracy"], color=COLOR_GREEN, marker="o", linewidth=1.9, label="Success rollout")
    ax1.plot(xs, failure["accuracy"], color=COLOR_ORANGE, marker="o", linewidth=1.9, label="Failure rollout")
    ax1.set_xticks(xs, bin_labels, rotation=35, ha="right")
    ax1.set_xlabel("Relative position in rollout")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.05)
    ax1.set_title(f"{cfg.title_prefix} All-Turn Estimation Accuracy")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    save(fig, out_dir, "figure2_all_turn_accuracy_by_relative_position")


def plot_figure3(cfg: DatasetConfig, est_df: pd.DataFrame, out_dir: Path, bin_labels: list[str]) -> None:
    overall = by_bin_mean(est_df, "reward", "reward", bin_labels)
    success = by_bin_mean(est_df[est_df["rollout_success"]], "reward", "reward", bin_labels)
    failure = by_bin_mean(est_df[~est_df["rollout_success"]], "reward", "reward", bin_labels)
    fig, ax1 = plt.subplots(figsize=(10.8, 5.2))
    ax2 = ax1.twinx()
    xs = np.arange(len(bin_labels))
    ax2.bar(xs, overall["count"], color="#d9d9d9", alpha=0.45, width=0.8, label="Samples")
    ax2.set_ylabel("Samples")
    ax2.set_ylim(0, max(overall["count"].max() * 1.35, 1))
    ax1.plot(xs, overall["reward"], color=COLOR_BLUE, marker="o", linewidth=2.3, label="Overall reward")
    ax1.plot(xs, success["reward"], color=COLOR_GREEN, marker="o", linewidth=1.9, label="Success rollout")
    ax1.plot(xs, failure["reward"], color=COLOR_ORANGE, marker="o", linewidth=1.9, label="Failure rollout")
    ax1.set_xticks(xs, bin_labels, rotation=35, ha="right")
    ax1.set_xlabel("Relative position in rollout")
    ax1.set_ylabel("Reward")
    ax1.set_ylim(0, 1.05)
    ax1.set_title(f"{cfg.title_prefix} Reward Curve")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    save(fig, out_dir, "figure3_reward_curve_by_relative_position")


def plot_figure4(cfg: DatasetConfig, est_df: pd.DataFrame, out_dir: Path, bin_labels: list[str]) -> None:
    success_ranges = est_df[
        est_df["rollout_success"] & est_df["pred_can_finish"] & est_df["pred_low"].notna() & est_df["pred_high"].notna()
    ].copy()
    success_ranges["range_class"] = success_ranges.apply(classify_range_direction, axis=1)
    success_ranges = success_ranges[success_ranges["range_class"] != "Other"].copy()
    counts = (
        success_ranges.groupby(["progress_bin", "range_class"], observed=False).size().unstack(fill_value=0).reindex(bin_labels).fillna(0)
    )
    count_order = ["Hit", "Overly optimistic", "Overly conservative"]
    for column in count_order:
        if column not in counts.columns:
            counts[column] = 0
    counts = counts[count_order]
    totals = counts.sum(axis=1)
    rates = counts.div(totals.replace(0, np.nan), axis=0)
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    xs = np.arange(len(bin_labels))
    bottom = np.zeros(len(bin_labels))
    colors = {"Hit": COLOR_GREEN, "Overly optimistic": COLOR_ORANGE, "Overly conservative": "#ffbb78"}
    for column in count_order:
        values = rates[column].fillna(0.0).to_numpy()
        ax.bar(xs, values, bottom=bottom, color=colors[column], width=0.78, label=column)
        bottom += values
    ax.set_xticks(xs, bin_labels, rotation=35, ha="right")
    ax.set_xlabel("Relative position in rollout")
    ax.set_ylabel("Rate within success-rollout interval predictions")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{cfg.title_prefix} Hit Rate and Miss Direction")
    ax.legend(loc="upper right")
    annotate_count_bars(ax, xs, np.minimum(bottom + 0.025, 1.02), [f"n={int(value)}" if value else "" for value in totals])
    save(fig, out_dir, "figure4_hit_optimism_pessimism_in_success_rollouts")


def plot_figure5(cfg: DatasetConfig, est_df: pd.DataFrame, out_dir: Path, bin_labels: list[str]) -> None:
    success_ranges = est_df[
        est_df["rollout_success"] & est_df["pred_can_finish"] & est_df["pred_width"].notna()
    ].copy()
    summary = (
        success_ranges.groupby("progress_bin", observed=False)["pred_width"]
        .agg(median="median", q25=lambda values: values.quantile(0.25), q75=lambda values: values.quantile(0.75), count="size")
        .reindex(bin_labels)
        .reset_index()
    )
    fig, ax1 = plt.subplots(figsize=(10.8, 5.2))
    ax2 = ax1.twinx()
    xs = np.arange(len(bin_labels))
    ax2.bar(xs, summary["count"].fillna(0), color="#d9d9d9", alpha=0.35, width=0.8, label="Samples")
    ax2.set_ylabel("Samples")
    ax2.set_ylim(0, max(summary["count"].fillna(0).max() * 1.35, 1))
    median = summary["median"].to_numpy(dtype=float)
    q25 = summary["q25"].to_numpy(dtype=float)
    q75 = summary["q75"].to_numpy(dtype=float)
    finite_q75 = q75[np.isfinite(q75)]
    y_max = max(float(finite_q75.max()) * 1.15, 1.0) if finite_q75.size else 1.0
    ax1.fill_between(xs, q25, q75, color="#9ecae1", alpha=0.4, label="IQR")
    ax1.plot(xs, median, color=COLOR_BLUE, marker="o", linewidth=2.3, label="Median width")
    ax1.set_xticks(xs, bin_labels, rotation=35, ha="right")
    ax1.set_xlabel("Relative position in rollout")
    ax1.set_ylabel("Predicted range width")
    ax1.set_ylim(0, y_max)
    ax1.set_title(f"{cfg.title_prefix} Range Width Change")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    save(fig, out_dir, "figure5_range_width_change_in_success_rollouts")


def plot_figure6(cfg: DatasetConfig, rollout_df: pd.DataFrame, est_df: pd.DataFrame, out_dir: Path) -> None:
    categories = ["Rollout", "Estimation"]
    total_inputs = [float(rollout_df["input_tokens_sum"].sum()), float(est_df["est_input_tokens"].dropna().sum())]
    cached_inputs = [float(rollout_df["cached_tokens_sum"].sum()), float(est_df["est_cached_tokens"].dropna().sum())]
    uncached_inputs = [total - cached for total, cached in zip(total_inputs, cached_inputs)]
    cached_rates = [cached / total if total else 0.0 for total, cached in zip(total_inputs, cached_inputs)]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9))
    axes[0].bar(categories, uncached_inputs, color="#3182bd", width=0.6, label="Uncached input tokens")
    axes[0].bar(categories, cached_inputs, bottom=uncached_inputs, color="#9ecae1", width=0.6, label="Cached input tokens")
    axes[0].set_ylabel("Input tokens")
    axes[0].set_title(f"{cfg.title_prefix} Cached vs. Total Input Tokens")
    axes[0].legend(loc="upper right")
    max_total = max(total_inputs) if total_inputs else 0.0
    for idx, (cached, total) in enumerate(zip(cached_inputs, total_inputs)):
        axes[0].text(idx, total + max_total * 0.02 if max_total else total, f"{cached/1000:.1f}k cached\n{total/1000:.1f}k total", ha="center", fontsize=9)
    axes[1].bar(categories, cached_rates, color=["#4c78a8", "#f58518"], width=0.55)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_ylabel("Cached share of input tokens")
    axes[1].set_title(f"{cfg.title_prefix} Cached Input Token Share")
    for idx, rate in enumerate(cached_rates):
        axes[1].text(idx, rate + 0.03, f"{rate:.1%}", ha="center")
    save(fig, out_dir, "figure6_cached_tokens_rollout_vs_estimation")


def plot_figure7(cfg: DatasetConfig, turn_df: pd.DataFrame, out_dir: Path, bin_labels: list[str]) -> None:
    overall = by_bin_mean(turn_df, "turn_incremental_tokens", "avg_tokens", bin_labels)
    success = by_bin_mean(turn_df[turn_df["rollout_success"]], "turn_incremental_tokens", "avg_tokens", bin_labels)
    failure = by_bin_mean(turn_df[~turn_df["rollout_success"]], "turn_incremental_tokens", "avg_tokens", bin_labels)
    fig, ax1 = plt.subplots(figsize=(10.8, 5.2))
    ax2 = ax1.twinx()
    xs = np.arange(len(bin_labels))
    ax2.bar(xs, overall["count"], color="#d9d9d9", alpha=0.45, width=0.8, label="Turn samples")
    ax2.set_ylabel("Turn samples")
    ax2.set_ylim(0, max(overall["count"].max() * 1.35, 1))
    ax1.plot(xs, overall["avg_tokens"], color=COLOR_BLUE, marker="o", linewidth=2.3, label="Overall")
    ax1.plot(xs, success["avg_tokens"], color=COLOR_GREEN, marker="o", linewidth=1.9, label="Success rollout")
    ax1.plot(xs, failure["avg_tokens"], color=COLOR_ORANGE, marker="o", linewidth=1.9, label="Failure rollout")
    ax1.set_xticks(xs, bin_labels, rotation=35, ha="right")
    ax1.set_xlabel("Relative position in rollout")
    ax1.set_ylabel("Average input+output tokens per turn (excluding history)")
    ax1.set_ylim(bottom=0)
    ax1.set_title(f"{cfg.title_prefix} Average Incremental Tokens per Turn")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    save(fig, out_dir, "figure7_average_tokens_used_in_rollout_turns")


def write_summary(cfg: DatasetConfig, rollout_df: pd.DataFrame, est_df: pd.DataFrame, out_dir: Path) -> None:
    first_turn = est_df[est_df["turn_idx"] == 1]
    success_ranges = est_df[
        est_df["rollout_success"] & est_df["pred_can_finish"] & est_df["pred_low"].notna() & est_df["pred_high"].notna()
    ].copy()
    success_ranges["range_class"] = success_ranges.apply(classify_range_direction, axis=1)
    rollout_cached_total = rollout_df["cached_tokens_sum"].sum()
    rollout_input_total = rollout_df["input_tokens_sum"].sum()
    est_cached_total = est_df["est_cached_tokens"].dropna().sum()
    est_input_total = est_df["est_input_tokens"].dropna().sum()
    lines = [
        f"# {cfg.task_display} {cfg.model_display} Figure Summary",
        "",
        f"- Rollouts: {len(rollout_df)}",
        f"- Successful rollouts: {int(rollout_df['success'].sum())}",
        f"- Failed rollouts: {int((~rollout_df['success']).sum())}",
        f"- Estimation samples: {len(est_df)}",
        f"- First-turn accuracy: {first_turn['can_finish_correct'].mean():.3f}",
        f"- Overall reward mean: {est_df['reward'].mean():.3f}",
    ]
    if cfg.reward_mode == "sokoban_custom":
        lines.append(f"- Stored reward mean: {est_df['stored_reward'].mean():.3f}")
    lines.extend(
        [
            f"- Success-rollout interval predictions: {len(success_ranges)}",
            f"- Hit count in success rollouts: {int((success_ranges['range_class'] == 'Hit').sum())}",
            f"- Overly optimistic count: {int((success_ranges['range_class'] == 'Overly optimistic').sum())}",
            f"- Overly conservative count: {int((success_ranges['range_class'] == 'Overly conservative').sum())}",
            f"- Rollout cached-input share: {(rollout_cached_total / rollout_input_total):.3f}" if rollout_input_total else "- Rollout cached-input share: n/a",
            f"- Estimation cached-input share: {(est_cached_total / est_input_total):.3f}" if est_input_total else "- Estimation cached-input share: n/a",
            "",
            f"- Rollout source: `{cfg.rollout_path}`",
            f"- Estimation source: `{cfg.estimation_path}`",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def write_combined_index(configs: list[DatasetConfig]) -> None:
    lines = [
        "# Agent Budget Control Figure Overview",
        "",
        "## Figure Sets",
        "",
    ]
    for cfg in configs:
        lines.append(f"- [{cfg.task_display} | {cfg.model_display}]({cfg.slug}/summary.md)")
    lines.append("")
    lines.append("Each folder contains `figure1` to `figure7` and a `summary.md` generated from `/u/ylin30/database/estimation` and the matching rollout source in `/u/ylin30/database/origin`.")
    (FIGURE_ROOT / "combined-figures.md").write_text("\n".join(lines) + "\n")


def format_pct(value: float) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{100.0 * float(value):.1f}%"


def format_float(value: float, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def compute_table1_rows(configs: list[DatasetConfig]) -> list[dict]:
    rows = []
    for cfg in configs:
        bin_edges = np.linspace(0.0, 1.0, cfg.progress_bins + 1)
        bin_labels = [f"{idx / cfg.progress_bins:.1f}-{(idx + 1) / cfg.progress_bins:.1f}" for idx in range(cfg.progress_bins)]
        reward_fn = compute_sokoban_reward if cfg.reward_mode == "sokoban_custom" else None
        rollout_df = build_rollout_df(load_json(cfg.rollout_path))
        est_df = build_estimation_df(load_json(cfg.estimation_path), bin_labels, bin_edges, reward_fn)

        first_turn = est_df[est_df["turn_idx"] == 1]
        success_est = est_df[est_df["rollout_success"]]
        fail_est = est_df[~est_df["rollout_success"]]

        success_interval_hit_rate = (
            success_est["contains_actual"].fillna(False).astype(bool).mean()
            if len(success_est)
            else np.nan
        )
        row = {
            "benchmark": cfg.task_display,
            "model": cfg.model_display,
            "slug": cfg.slug,
            "rollout_success_rate": rollout_df["success"].mean(),
            "rollout_avg_turns": rollout_df["total_turns"].mean(),
            "first_turn_success_pred_acc": first_turn["can_finish_correct"].mean() if len(first_turn) else np.nan,
            "estimation_total": len(est_df),
            "estimation_success_rollouts": int(success_est.shape[0]),
            "estimation_fail_rollouts": int(fail_est.shape[0]),
            "pred_hit_success_rollouts": success_est["can_finish_correct"].mean() if len(success_est) else np.nan,
            "pred_hit_fail_rollouts": fail_est["can_finish_correct"].mean() if len(fail_est) else np.nan,
            "success_rollout_interval_hit_rate": success_interval_hit_rate,
            "success_rollout_custom_reward": success_est["custom_reward"].mean() if len(success_est) else np.nan,
        }
        rows.append(row)
    return rows


def write_table1(configs: list[DatasetConfig]) -> None:
    OVERALL_DIR.mkdir(parents=True, exist_ok=True)
    rows = compute_table1_rows(configs)
    df = pd.DataFrame(rows)

    best_indices = compute_table1_best_indices(df)

    csv_df = df.copy()
    csv_df.to_csv(OVERALL_DIR / "table1.csv", index=False)

    lines = [
        "# Table 1",
        "",
        "Within each benchmark, the best value in each metric column is marked in red.",
        "`Avg Turns` uses lower-is-better; other highlighted numeric columns use higher-is-better. `Estimations (S/F)` is highlighted by total estimation count.",
        "",
        "| Benchmark | Model | Rollout Success | Avg Turns | First-Turn Success Pred Acc | Estimations (Succ/Fail Rollouts) | Pred Hit on Success Rollouts | Pred Hit on Fail Rollouts | Success-Rollout Interval Hit | Success-Rollout Reward |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for idx, row in df.sort_values(["benchmark", "rollout_success_rate"], ascending=[True, False]).iterrows():
        def maybe_red(metric_key: str, text: str) -> str:
            return f"<span style=\"color:red\">{text}</span>" if idx in best_indices[metric_key] else text

        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["benchmark"]),
                    str(row["model"]),
                    maybe_red("rollout_success_rate", format_pct(row["rollout_success_rate"])),
                    maybe_red("rollout_avg_turns", format_float(row["rollout_avg_turns"])),
                    maybe_red("first_turn_success_pred_acc", format_pct(row["first_turn_success_pred_acc"])),
                    maybe_red(
                        "estimation_total",
                        f"{int(row['estimation_total'])} ({int(row['estimation_success_rollouts'])}/{int(row['estimation_fail_rollouts'])})",
                    ),
                    maybe_red("pred_hit_success_rollouts", format_pct(row["pred_hit_success_rollouts"])),
                    maybe_red("pred_hit_fail_rollouts", format_pct(row["pred_hit_fail_rollouts"])),
                    maybe_red("success_rollout_interval_hit_rate", format_pct(row["success_rollout_interval_hit_rate"])),
                    maybe_red("success_rollout_custom_reward", format_float(row["success_rollout_custom_reward"], 3)),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "Columns:",
            "- `Estimations (Succ/Fail Rollouts)` means total estimation samples, followed by counts from successful and failed rollouts.",
            "- `Pred Hit on Success/Fail Rollouts` is the mean of `can_finish_correct` over all estimation samples in that rollout group.",
            "- `Success-Rollout Interval Hit` counts a success-rollout sample as a hit only when the predicted interval contains the actual remaining token target; missing intervals count as misses.",
            "- `Success-Rollout Reward` uses the custom score requested in `figure-plan.md`, averaged over success-rollout estimation samples.",
        ]
    )

    (OVERALL_DIR / "table1.md").write_text("\n".join(lines) + "\n")
    render_table1_figure(df, best_indices)


def compute_table1_best_indices(df: pd.DataFrame) -> dict[str, set[int]]:
    metric_directions = {
        "rollout_success_rate": "max",
        "rollout_avg_turns": "min",
        "first_turn_success_pred_acc": "max",
        "estimation_total": "max",
        "pred_hit_success_rollouts": "max",
        "pred_hit_fail_rollouts": "max",
        "success_rollout_interval_hit_rate": "max",
        "success_rollout_custom_reward": "max",
    }
    best_indices = {metric: set() for metric in metric_directions}
    for _, group in df.groupby("benchmark", sort=False):
        for metric, direction in metric_directions.items():
            series = group[metric]
            target = series.min() if direction == "min" else series.max()
            winners = group[series == target].index.tolist()
            best_indices[metric].update(winners)
    return best_indices


def render_table1_figure(df: pd.DataFrame, best_indices: dict[str, set[int]]) -> None:
    display_df = df.sort_values(["benchmark", "rollout_success_rate"], ascending=[True, False]).copy()
    display_df["rollout_success_rate"] = display_df["rollout_success_rate"].map(format_pct)
    display_df["rollout_avg_turns"] = display_df["rollout_avg_turns"].map(format_float)
    display_df["first_turn_success_pred_acc"] = display_df["first_turn_success_pred_acc"].map(format_pct)
    display_df["estimations"] = display_df.apply(
        lambda row: f"{int(row['estimation_total'])} ({int(row['estimation_success_rollouts'])}/{int(row['estimation_fail_rollouts'])})",
        axis=1,
    )
    display_df["pred_hit_success_rollouts"] = display_df["pred_hit_success_rollouts"].map(format_pct)
    display_df["pred_hit_fail_rollouts"] = display_df["pred_hit_fail_rollouts"].map(format_pct)
    display_df["success_rollout_interval_hit_rate"] = display_df["success_rollout_interval_hit_rate"].map(format_pct)
    display_df["success_rollout_custom_reward"] = display_df["success_rollout_custom_reward"].map(lambda v: format_float(v, 3))

    headers = [
        "Benchmark",
        "Model",
        "Rollout\nSuccess",
        "Avg\nTurns",
        "First-Turn\nPred Acc",
        "Estimations\n(S/F)",
        "Pred Hit\nSuccess",
        "Pred Hit\nFail",
        "Success\nInterval Hit",
        "Success\nReward",
    ]
    columns = [
        "benchmark",
        "model",
        "rollout_success_rate",
        "rollout_avg_turns",
        "first_turn_success_pred_acc",
        "estimations",
        "pred_hit_success_rollouts",
        "pred_hit_fail_rollouts",
        "success_rollout_interval_hit_rate",
        "success_rollout_custom_reward",
    ]
    cell_text = display_df[columns].values.tolist()

    fig_height = 1.6 + 0.55 * len(cell_text)
    fig, ax = plt.subplots(figsize=(19, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.35)

    header_color = "#dbeafe"
    row_even = "#f8fafc"
    row_odd = "#ffffff"
    edge_color = "#cbd5e1"
    sota_color = "#dc2626"

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(edge_color)
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(weight="bold", color="#111827")
        else:
            cell.set_facecolor(row_even if row % 2 == 0 else row_odd)
            if col in (0, 1):
                cell.set_text_props(ha="left")

    metric_to_col = {
        "rollout_success_rate": 2,
        "rollout_avg_turns": 3,
        "first_turn_success_pred_acc": 4,
        "estimation_total": 5,
        "pred_hit_success_rollouts": 6,
        "pred_hit_fail_rollouts": 7,
        "success_rollout_interval_hit_rate": 8,
        "success_rollout_custom_reward": 9,
    }
    sorted_indices = display_df.index.tolist()
    for table_row, df_index in enumerate(sorted_indices, start=1):
        for metric_key, col_idx in metric_to_col.items():
            if df_index in best_indices[metric_key]:
                table[(table_row, col_idx)].get_text().set_color(sota_color)
                table[(table_row, col_idx)].get_text().set_weight("bold")

    ax.set_title("Table 1. Overall Benchmark Comparison", fontsize=16, fontweight="bold", pad=12)
    fig.savefig(OVERALL_DIR / "table1.png", bbox_inches="tight", dpi=320)
    plt.close(fig)


def generate_dataset(cfg: DatasetConfig) -> None:
    bin_edges = np.linspace(0.0, 1.0, cfg.progress_bins + 1)
    bin_labels = [f"{idx / cfg.progress_bins:.1f}-{(idx + 1) / cfg.progress_bins:.1f}" for idx in range(cfg.progress_bins)]
    reward_fn = compute_sokoban_reward if cfg.reward_mode == "sokoban_custom" else None

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    rollout_payload = load_json(cfg.rollout_path)
    estimation_payload = load_json(cfg.estimation_path)

    rollout_df = build_rollout_df(rollout_payload)
    turn_df = build_rollout_turn_df(rollout_payload, bin_labels, bin_edges)
    est_df = build_estimation_df(estimation_payload, bin_labels, bin_edges, reward_fn)

    plot_figure1(cfg, est_df, cfg.out_dir)
    plot_figure2(cfg, est_df, cfg.out_dir, bin_labels)
    plot_figure3(cfg, est_df, cfg.out_dir, bin_labels)
    plot_figure4(cfg, est_df, cfg.out_dir, bin_labels)
    plot_figure5(cfg, est_df, cfg.out_dir, bin_labels)
    plot_figure6(cfg, rollout_df, est_df, cfg.out_dir)
    plot_figure7(cfg, turn_df, cfg.out_dir, bin_labels)
    write_summary(cfg, rollout_df, est_df, cfg.out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate figure sets for budget-control experiments.")
    parser.add_argument("--slug", choices=sorted(CONFIG_BY_SLUG.keys()), default=None, help="Generate only one figure set.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.slug:
        configs = [CONFIG_BY_SLUG[args.slug]]
    else:
        configs = CONFIGS

    for cfg in configs:
        generate_dataset(cfg)

    if not args.slug:
        write_combined_index(CONFIGS)
        write_table1(OVERALL_CONFIGS)


if __name__ == "__main__":
    main()
