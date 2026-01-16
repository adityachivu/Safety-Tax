#!/usr/bin/env python3
"""
Run lm-evaluation-harness benchmarks against a Modal vLLM endpoint.

Usage:
    python run_eval.py --endpoint https://modal-labs-civicmachines--vllm-inference-server-serve.modal.run --tasks mmlu,hellaswag
    python run_eval.py --endpoint https://modal-labs-civicmachines--vllm-inference-server-serve.modal.run --tasks aime24_nofigures --output_path ./results
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Run lm-evaluation-harness benchmarks against a Modal vLLM endpoint"
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        required=True,
        help="Modal endpoint URL (e.g., https://modal-labs-civicmachines--vllm-inference-server-serve.modal.run)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B-Thinking-2507-FP8",
        help="Model name for tokenizer and API requests (default: Qwen/Qwen2.5-32B-Instruct)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        required=True,
        help="Comma-separated list of tasks to evaluate (e.g., mmlu,hellaswag,aime24_nofigures)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./eval_results",
        help="Directory to save evaluation results (default: ./eval_results)",
    )
    parser.add_argument(
        "--limit",
        type=str,
        default=None,
        help="Limit number of examples: integer (e.g., 100) or float for percentage (e.g., 0.1 for 10%%)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Max tokens for generation (default: 2048)",
    )

    args = parser.parse_args()

    # Build base_url from endpoint (strip trailing slash, append /v1/chat/completions)
    base_url = args.endpoint.rstrip("/") + "/v1/chat/completions"

    # Build model_args with user-provided model and constructed endpoint
    model_args = ",".join([
        f"model={args.model}",
        f"base_url={base_url}",
        "tokenizer_backend=huggingface",
        f"tokenizer={args.model}",
    ])

    # Build lm_eval command
    cmd = [
        "lm_eval",
        "--model", "local-chat-completions",
        "--model_args", model_args,
        "--tasks", args.tasks,
        "--apply_chat_template",
        "--output_path", args.output_path,
        "--log_samples",
    ]

    if args.limit:
        cmd.extend(["--limit", args.limit])

    cmd.extend(["--gen_kwargs", f"max_tokens={args.max_tokens}"])

    print(f"Running: {' '.join(cmd)}")
    
    # Execute lm_eval
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
