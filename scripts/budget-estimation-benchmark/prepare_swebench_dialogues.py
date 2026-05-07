#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _safe_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_text_from_blocks(blocks: Iterable[Any]) -> str:
    parts: List[str] = []
    for block in blocks:
        if isinstance(block, dict):
            block_type = str(block.get("type", "") or "")
            if block_type in {"output_text", "input_text", "text"}:
                text = block.get("text")
                if text is not None:
                    parts.append(str(text))
        elif block is not None:
            parts.append(str(block))
    return "\n".join(part for part in parts if part).strip()


def _serialize_tool_call(name: str, arguments: Any) -> str:
    arguments_text = "" if arguments is None else str(arguments)
    return f'<tool_call name="{name}">{arguments_text}</tool_call>'


def _extract_response_output(message: Dict[str, Any]) -> Tuple[str, List[str]]:
    content_parts: List[str] = []
    action_names: List[str] = []
    for item in list(message.get("output") or []):
        item_type = str(item.get("type", "") or "")
        if item_type == "message":
            text = _extract_text_from_blocks(list(item.get("content") or []))
            if text:
                content_parts.append(text)
        elif item_type == "function_call":
            name = str(item.get("name", "") or "tool")
            arguments = item.get("arguments")
            action_names.append(name)
            content_parts.append(_serialize_tool_call(name, arguments))
    return "\n".join(part for part in content_parts if part).strip(), action_names


def _extract_chat_tool_calls(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    tool_calls = list(message.get("tool_calls") or [])
    function_call = message.get("function_call")
    if function_call:
        tool_calls.append(
            {
                "function": function_call,
                "type": "function",
            }
        )
    return tool_calls


def _extract_chat_output(message: Dict[str, Any]) -> Tuple[str, List[str]]:
    content_parts: List[str] = []
    action_names: List[str] = []
    content = message.get("content")
    if isinstance(content, str):
        normalized = content.strip()
        if normalized:
            content_parts.append(normalized)
    elif isinstance(content, list):
        text = _extract_text_from_blocks(content)
        if text:
            content_parts.append(text)

    for tool_call in _extract_chat_tool_calls(message):
        function = tool_call.get("function") or {}
        name = str(function.get("name", "") or "tool")
        arguments = function.get("arguments")
        action_names.append(name)
        content_parts.append(_serialize_tool_call(name, arguments))

    return "\n".join(part for part in content_parts if part).strip(), action_names


def _extract_usage(usage: Dict[str, Any]) -> Dict[str, Optional[int]]:
    input_tokens = _safe_int(usage.get("input_tokens", usage.get("prompt_tokens")))
    output_tokens = _safe_int(usage.get("output_tokens", usage.get("completion_tokens")))
    total_tokens = _safe_int(usage.get("total_tokens"))
    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = int(input_tokens or 0) + int(output_tokens or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _is_response_message(message: Dict[str, Any]) -> bool:
    return str(message.get("object", "") or "") == "response"


def _is_assistant_message(message: Dict[str, Any]) -> bool:
    return str(message.get("role", "") or "") == "assistant"


def _is_tool_output_message(message: Dict[str, Any]) -> bool:
    return (
        str(message.get("type", "") or "") == "function_call_output"
        or str(message.get("role", "") or "") == "tool"
    )


def _is_user_message(message: Dict[str, Any]) -> bool:
    return str(message.get("role", "") or "") == "user"


def _is_exit_message(message: Dict[str, Any]) -> bool:
    return str(message.get("role", "") or "") == "exit"


def _extract_tool_output_text(message: Dict[str, Any]) -> str:
    if str(message.get("type", "") or "") == "function_call_output":
        return str(message.get("output", "") or "").strip()
    return str(message.get("content", "") or "").strip()


def _extract_assistant_turn(message: Dict[str, Any]) -> Tuple[str, List[str], Dict[str, Optional[int]]]:
    if _is_response_message(message):
        content, action_names = _extract_response_output(message)
        usage = _extract_usage(dict(message.get("usage") or {}))
        return content, action_names, usage
    if _is_assistant_message(message):
        content, action_names = _extract_chat_output(message)
        response = ((message.get("extra") or {}).get("response") or {})
        usage = _extract_usage(dict(response.get("usage") or {}))
        return content, action_names, usage
    raise ValueError("Message is not an assistant turn")


def _extract_initial_system_and_user(messages: List[Dict[str, Any]]) -> Tuple[str, str, int]:
    system_prompt = ""
    user_prompt = ""
    cursor = 0
    if cursor < len(messages) and str(messages[cursor].get("role", "") or "") == "system":
        system_prompt = str(messages[cursor].get("content", "") or "")
        cursor += 1
    if cursor < len(messages) and str(messages[cursor].get("role", "") or "") == "user":
        user_prompt = str(messages[cursor].get("content", "") or "")
        cursor += 1
    return system_prompt, user_prompt, cursor


def _is_submitted_exit(exit_status: Optional[str]) -> bool:
    normalized = str(exit_status or "").strip().lower()
    return normalized in {"submitted", "complete", "completed", "success", "succeeded"}


def _collect_traj_files(input_dir: Path) -> List[Path]:
    input_dir = input_dir.expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if input_dir.is_file():
        raise ValueError(f"Expected a directory, got file: {input_dir}")
    traj_files = sorted(path for path in input_dir.rglob("*.traj.json") if path.is_file())
    if not traj_files:
        raise ValueError(f"Could not find any *.traj.json under {input_dir}")
    return traj_files


def _load_preds(input_dir: Path) -> Dict[str, Any]:
    preds: Dict[str, Any] = {}
    for candidate in sorted(path for path in input_dir.rglob("preds.json") if path.is_file()):
        with candidate.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            preds.update(payload)
    return preds


def _load_eval_summary(input_dir: Path) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "has_resolution_data": False,
        "submitted_ids": set(),
        "completed_ids": set(),
        "resolved_ids": set(),
        "unresolved_ids": set(),
        "incomplete_ids": set(),
        "empty_patch_ids": set(),
        "error_ids": set(),
        "source_files": [],
    }

    candidate_files = sorted(
        path
        for path in input_dir.glob("*.json")
        if path.is_file() and path.name != "preds.json"
    )
    for candidate in candidate_files:
        with candidate.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            continue
        if "resolved_ids" not in payload:
            continue

        summary["has_resolution_data"] = True
        summary["source_files"].append(str(candidate))
        for field in (
            "submitted_ids",
            "completed_ids",
            "resolved_ids",
            "unresolved_ids",
            "incomplete_ids",
            "empty_patch_ids",
            "error_ids",
        ):
            values = payload.get(field) or []
            if isinstance(values, list):
                summary[field].update(str(value) for value in values if value is not None)

    return summary


def _extract_interstitial_text(message: Dict[str, Any]) -> str:
    if _is_tool_output_message(message):
        return _extract_tool_output_text(message)
    if _is_user_message(message):
        return str(message.get("content", "") or "").strip()
    return ""


def _convert_traj_to_rollout(
    traj_path: Path,
    env_index: int,
    preds: Dict[str, Any],
    eval_summary: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    with traj_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    messages = list(payload.get("messages") or [])
    if not messages:
        return None

    system_prompt, current_user_prompt, cursor = _extract_initial_system_and_user(messages)
    instance_id = str(payload.get("instance_id") or traj_path.stem)
    turns: List[Dict[str, Any]] = []

    while cursor < len(messages):
        message = messages[cursor]
        if _is_exit_message(message):
            break
        if not (_is_response_message(message) or _is_assistant_message(message)):
            cursor += 1
            continue

        raw_response, action_names, usage = _extract_assistant_turn(message)
        turn_messages: List[Dict[str, str]] = []
        if system_prompt:
            turn_messages.append({"role": "system", "content": system_prompt})
        turn_messages.append({"role": "user", "content": current_user_prompt})

        turns.append(
            {
                "turn_idx": len(turns) + 1,
                "messages": turn_messages,
                "user_prompt": current_user_prompt,
                "raw_response": raw_response,
                "api_interaction_count": 1,
                "api_input_tokens": usage["input_tokens"],
                "api_output_tokens": usage["output_tokens"],
                "api_total_tokens": usage["total_tokens"],
                "action_names": list(action_names),
                "actions": list(action_names),
                "toolcalls_used": len(action_names),
                "success": False,
            }
        )

        cursor += 1
        interstitial_parts: List[str] = []
        while cursor < len(messages):
            next_message = messages[cursor]
            if _is_exit_message(next_message) or _is_response_message(next_message) or _is_assistant_message(next_message):
                break
            interstitial_text = _extract_interstitial_text(next_message)
            if interstitial_text:
                interstitial_parts.append(interstitial_text)
            cursor += 1
        current_user_prompt = "\n\n".join(interstitial_parts).strip()

    if not turns:
        return None

    exit_message = next((message for message in reversed(messages) if _is_exit_message(message)), None)
    exit_extra = (exit_message or {}).get("extra") or {}
    exit_status = exit_extra.get("exit_status")
    submitted_patch = exit_extra.get("submission")
    pred_entry = preds.get(instance_id)
    pred_patch = pred_entry.get("model_patch") if isinstance(pred_entry, dict) else None
    summary = eval_summary or {}
    has_resolution_data = bool(summary.get("has_resolution_data"))
    summary_submitted_ids = summary.get("submitted_ids") or set()
    summary_completed_ids = summary.get("completed_ids") or set()
    summary_resolved_ids = summary.get("resolved_ids") or set()
    summary_unresolved_ids = summary.get("unresolved_ids") or set()
    summary_incomplete_ids = summary.get("incomplete_ids") or set()
    summary_empty_patch_ids = summary.get("empty_patch_ids") or set()
    summary_error_ids = summary.get("error_ids") or set()

    rollout_submitted = (
        instance_id in summary_submitted_ids
        if has_resolution_data
        else _is_submitted_exit(exit_status)
    )
    rollout_completed = (
        instance_id in summary_completed_ids
        if has_resolution_data
        else rollout_submitted
    )
    rollout_resolved = (
        instance_id in summary_resolved_ids
        if has_resolution_data
        else rollout_submitted
    )
    rollout_success = bool(rollout_resolved)
    turns[-1]["success"] = rollout_success

    total_input_tokens = sum(int(turn.get("api_input_tokens") or 0) for turn in turns)
    total_output_tokens = sum(int(turn.get("api_output_tokens") or 0) for turn in turns)
    total_tokens = sum(int(turn.get("api_total_tokens") or 0) for turn in turns)

    return {
        "env_id": env_index,
        "absolute_env_id": env_index,
        "instance_id": instance_id,
        "mode": "dialogue",
        "tag": "SWEBench",
        "api_interaction_count": len(turns),
        "api_input_tokens": total_input_tokens,
        "api_output_tokens": total_output_tokens,
        "api_total_tokens": total_tokens,
        "rollout_success": rollout_success,
        "rollout_submitted": bool(rollout_submitted),
        "rollout_completed": bool(rollout_completed),
        "rollout_resolved": bool(rollout_resolved),
        "rollout_unresolved": bool(instance_id in summary_unresolved_ids) if has_resolution_data else (not rollout_success),
        "rollout_incomplete": bool(instance_id in summary_incomplete_ids) if has_resolution_data else False,
        "rollout_empty_patch": bool(instance_id in summary_empty_patch_ids) if has_resolution_data else False,
        "rollout_error": bool(instance_id in summary_error_ids) if has_resolution_data else False,
        "exit_status": exit_status,
        "submission_patch": submitted_patch or pred_patch,
        "turns": turns,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert SWE-bench origin traj.json files into token-estimation rollout dialogues."
    )
    parser.add_argument("--input-dir", required=True, help="SWE-bench origin directory containing *.traj.json files")
    parser.add_argument("--output-json", required=True, help="Path to converted rollout json")
    parser.add_argument("--max-rollouts", type=int, default=None, help="Optional cap for debugging")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    traj_files = _collect_traj_files(input_dir)
    preds = _load_preds(input_dir.expanduser().resolve())
    eval_summary = _load_eval_summary(input_dir.expanduser().resolve())
    if args.max_rollouts is not None:
        traj_files = traj_files[: int(args.max_rollouts)]

    rollouts: List[Dict[str, Any]] = []
    skipped = 0
    for env_index, traj_path in enumerate(traj_files):
        rollout = _convert_traj_to_rollout(
            traj_path,
            env_index=env_index,
            preds=preds,
            eval_summary=eval_summary,
        )
        if rollout is None:
            skipped += 1
            continue
        rollouts.append(rollout)

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(rollouts, handle, ensure_ascii=False, indent=2)

    print(f"Converted {len(rollouts)} rollouts from {input_dir.expanduser().resolve()}")
    if eval_summary.get("has_resolution_data"):
        print(
            "Loaded SWE-bench resolution summary from "
            + ", ".join(eval_summary.get("source_files") or [])
        )
    if skipped:
        print(f"Skipped {skipped} rollout(s) with no assistant turns")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
