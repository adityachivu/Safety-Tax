#!/usr/bin/env python3
"""
Run lm-evaluation-harness benchmarks against a Modal vLLM endpoint.

Usage:
    python run_eval.py --endpoint https://modal-labs-civicmachines--vllm-inference-server-serve.modal.run --tasks mmlu,hellaswag
    python run_eval.py --endpoint https://modal-labs-civicmachines--vllm-inference-server-serve.modal.run --tasks aime24_nofigures --output_path ./results
"""

import argparse
import os
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
        default="ArliAI/gpt-oss-20b-Derestricted",
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
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 for greedy decoding)",
    )
    parser.add_argument(
        "--no_chat_template",
        action="store_true",
        help="Disable --apply_chat_template (use for non-chat tasks like hellaswag)",
    )
    parser.add_argument(
        "--completions",
        action="store_true",
        help="Use /v1/completions endpoint (supports logprobs for hellaswag, arc, etc.) instead of /v1/chat/completions",
    )

    args = parser.parse_args()

    # Build base_url from endpoint
    if args.completions:
        base_url = args.endpoint.rstrip("/") + "/v1/completions"
        model_type = "local-completions"
    else:
        base_url = args.endpoint.rstrip("/") + "/v1/chat/completions"
        model_type = "local-chat-completions"

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
        "--model", model_type,
        "--model_args", model_args,
        "--tasks", args.tasks,
        "--output_path", args.output_path,
        "--log_samples",
    ]

    if not args.no_chat_template:
        cmd.append("--apply_chat_template")

    if args.limit:
        cmd.extend(["--limit", args.limit])

    # Build generation kwargs with max_tokens and temperature
    gen_kwargs = f"max_tokens={args.max_tokens},temperature={args.temperature}"
    cmd.extend(["--gen_kwargs", gen_kwargs])

    print(f"Running: {' '.join(cmd)}")
    
    # Set EXTRACTION_ENDPOINT for answer extraction in MATH/GPQA/AIME tasks
    env = os.environ.copy()
    env["EXTRACTION_ENDPOINT"] = args.endpoint.rstrip("/")
    
    # Execute lm_eval
    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
