"""Standalone runnable entrypoint for HR Policy Assistant."""

from __future__ import annotations

import argparse
from pathlib import Path

from forgeai.runtime import AgentRuntime


def main() -> int:
    parser = argparse.ArgumentParser(description="Run HR Policy Assistant")
    parser.add_argument("--message", "-m", default=None, help="Single question to ask (omit for interactive chat).")
    parser.add_argument("--use-llm", action="store_true", help="Force LLM mode using configured API credentials.")
    args = parser.parse_args()

    agent_path = Path(__file__).resolve().parent / "agent.json"
    runtime = AgentRuntime.from_file(agent_path)
    use_llm = True if args.use_llm else False

    if args.message:
        reply = runtime.chat(args.message, use_llm=use_llm)
        print(reply.answer)
        return 0

    print(f"Loaded {runtime.agent_def.agent_name}. Type 'exit' to quit.")
    while True:
        user_msg = input("You> ").strip()
        if user_msg.lower() in {"exit", "quit"}:
            break
        reply = runtime.chat(user_msg, use_llm=use_llm)
        print(f"\nAgent> {reply.answer}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
