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


def _resolve_traj_dir(input_dir: Path) -> Tuple[Path, str]:
    input_dir = input_dir.expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if input_dir.is_file():
        raise ValueError(f"Expected a directory, got file: {input_dir}")
    if (input_dir / "trajs").is_dir():
        return input_dir / "trajs", input_dir.name
    if input_dir.name == "trajs":
        return input_dir, input_dir.parent.name
    raise ValueError(
        f"Could not find trajs/ under {input_dir}. Pass a mini-SWE-bench model directory or its trajs/ subdirectory."
    )


def _maybe_load_preds_json(input_dir: Path, traj_dir: Path) -> Dict[str, Any]:
    candidates = [
        traj_dir / "preds.json",
        input_dir / "preds.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                return payload
    return {}


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
        content = content.strip()
        if content:
            content_parts.append(content)
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


def _is_response_message(message: Dict[str, Any]) -> bool:
    return str(message.get("object", "") or "") == "response"


def _is_assistant_message(message: Dict[str, Any]) -> bool:
    return str(message.get("role", "") or "") == "assistant"


def _is_tool_output_message(message: Dict[str, Any]) -> bool:
    return (
        str(message.get("type", "") or "") == "function_call_output"
        or str(message.get("role", "") or "") == "tool"
    )


def _is_exit_message(message: Dict[str, Any]) -> bool:
    return str(message.get("role", "") or "") == "exit"


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


def _extract_tool_output_text(message: Dict[str, Any]) -> str:
    if str(message.get("type", "") or "") == "function_call_output":
        return str(message.get("output", "") or "").strip()
    return str(message.get("content", "") or "").strip()


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


def _convert_traj_to_rollout(
    traj_path: Path,
    env_index: int,
    preds: Dict[str, Any],
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
        tool_outputs: List[str] = []
        while cursor < len(messages):
            next_message = messages[cursor]
            if _is_exit_message(next_message) or _is_response_message(next_message) or _is_assistant_message(next_message):
                break
            if _is_tool_output_message(next_message):
                output_text = _extract_tool_output_text(next_message)
                if output_text:
                    tool_outputs.append(output_text)
            cursor += 1
        current_user_prompt = "\n\n".join(tool_outputs).strip()

    if not turns:
        return None

    exit_message = next((message for message in reversed(messages) if _is_exit_message(message)), None)
    exit_extra = (exit_message or {}).get("extra") or {}
    exit_status = exit_extra.get("exit_status")
    submitted_patch = exit_extra.get("submission")
    pred_entry = preds.get(instance_id)
    pred_patch = pred_entry.get("model_patch") if isinstance(pred_entry, dict) else None
    rollout_success = _is_submitted_exit(exit_status)
    turns[-1]["success"] = rollout_success

    total_input_tokens = sum(int(turn.get("api_input_tokens") or 0) for turn in turns)
    total_output_tokens = sum(int(turn.get("api_output_tokens") or 0) for turn in turns)
    total_tokens = sum(int(turn.get("api_total_tokens") or 0) for turn in turns)

    return {
        "env_id": env_index,
        "absolute_env_id": env_index,
        "instance_id": instance_id,
        "mode": "dialogue",
        "tag": "MiniSWEBench",
        "api_interaction_count": len(turns),
        "api_input_tokens": total_input_tokens,
        "api_output_tokens": total_output_tokens,
        "api_total_tokens": total_tokens,
        "rollout_success": rollout_success,
        "exit_status": exit_status,
        "submission_patch": submitted_patch or pred_patch,
        "turns": turns,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert mini-SWE-agent traj.json files into token-estimation rollout dialogues."
    )
    parser.add_argument("--input-dir", required=True, help="Mini-SWE-bench directory or its trajs/ subdirectory")
    parser.add_argument("--output-json", required=True, help="Path to converted rollout json")
    parser.add_argument("--max-rollouts", type=int, default=None, help="Optional cap for debugging")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    traj_dir, _ = _resolve_traj_dir(input_dir)
    preds = _maybe_load_preds_json(input_dir.resolve(), traj_dir)

    traj_files = sorted(traj_dir.glob("*/*.traj.json"))
    if args.max_rollouts is not None:
        traj_files = traj_files[: int(args.max_rollouts)]

    rollouts: List[Dict[str, Any]] = []
    skipped = 0
    for env_index, traj_path in enumerate(traj_files):
        rollout = _convert_traj_to_rollout(traj_path, env_index=env_index, preds=preds)
        if rollout is None:
            skipped += 1
            continue
        rollouts.append(rollout)

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(rollouts, handle, ensure_ascii=False, indent=2)

    print(f"Converted {len(rollouts)} rollouts from {traj_dir}")
    if skipped:
        print(f"Skipped {skipped} rollout(s) with no assistant turns")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
