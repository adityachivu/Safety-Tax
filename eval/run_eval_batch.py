#!/usr/bin/env python3
"""
Run multiple lm-evaluation-harness benchmarks sequentially from a JSON config file.

Usage:
    python run_eval_batch.py --config eval_config.json --endpoint https://modal-labs-civicmachines--vllm-inference-server-serve.modal.run
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_task(task_config, endpoint, output_path, model, script_dir):
    """Run a single evaluation task."""
    tasks = task_config.get("tasks")
    if not tasks:
        print(f"Skipping task config without 'tasks' field: {task_config}")
        return False
    
    # Build command with full path to run_eval.py
    run_eval_path = script_dir / "run_eval.py"
    cmd = [
        "python", str(run_eval_path),
        "--endpoint", endpoint,
        "--tasks", tasks,
        "--output_path", output_path,
        "--model", model,
    ]
    
    # Optional flags
    if task_config.get("completions", False):
        cmd.append("--completions")
    
    if task_config.get("no_chat_template", False):
        cmd.append("--no_chat_template")
    
    # Optional parameters
    if "max_tokens" in task_config:
        cmd.extend(["--max_tokens", str(task_config["max_tokens"])])
    
    if "temperature" in task_config:
        cmd.extend(["--temperature", str(task_config["temperature"])])
    
    if "limit" in task_config:
        cmd.extend(["--limit", str(task_config["limit"])])
    
    # Print and run
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\n❌ Task failed: {tasks}")
        return False
    
    print(f"\n✅ Task completed: {tasks}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple evaluation tasks from a JSON config file"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON config file with task specifications",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        required=True,
        help="Modal endpoint URL",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-oss-20b",
        help="Model name for tokenizer and API requests",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./eval_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop execution if any task fails",
    )

    args = parser.parse_args()

    # Get the directory where this script is located
    script_dir = Path(__file__).parent.resolve()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        config = json.load(f)

    if "tasks" not in config or not isinstance(config["tasks"], list):
        print("Error: Config must have a 'tasks' array")
        sys.exit(1)

    # Run each task
    total_tasks = len(config["tasks"])
    successful = 0
    failed = 0

    print(f"Loaded {total_tasks} tasks from {args.config}")

    for i, task_config in enumerate(config["tasks"], 1):
        print(f"\n[{i}/{total_tasks}] Processing task configuration...")
        
        success = run_task(
            task_config,
            args.endpoint,
            args.output_path,
            args.model,
            script_dir
        )
        
        if success:
            successful += 1
        else:
            failed += 1
            if args.stop_on_error:
                print("\n❌ Stopping due to error (--stop_on_error is set)")
                break

    # Summary
    print(f"\n{'='*80}")
    print(f"BATCH EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total tasks: {total_tasks}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"{'='*80}\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
