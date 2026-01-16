#!/usr/bin/env python3
"""
Test extraction functionality on existing sample responses without running full evaluation.

Usage:
    # For GPQA samples
    EXTRACTION_ENDPOINT=https://your-modal-endpoint.modal.run python test_extraction.py --samples eval_results/.../samples_gpqa_diamond_openai_*.jsonl --task gpqa
    
    # For MATH samples  
    EXTRACTION_ENDPOINT=https://your-modal-endpoint.modal.run python test_extraction.py --samples eval_results/.../samples_openai_math_*.jsonl --task math
    
    # For AIME samples
    EXTRACTION_ENDPOINT=https://your-modal-endpoint.modal.run python test_extraction.py --samples eval_results/.../samples_aime24_*.jsonl --task aime
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add lm-evaluation-harness to path
sys.path.insert(0, str(Path(__file__).parent / "lm-evaluation-harness"))

from lm_eval.tasks._extraction_utils import get_extraction_sampler


def load_samples(samples_file):
    """Load samples from JSONL file."""
    samples = []
    with open(samples_file, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def test_gpqa_extraction(samples):
    """Test GPQA extraction."""
    from lm_eval.tasks.gpqa.openai.utils import extract_answer, EXTRACTION_TEMPLATE, QUERY_TEMPLATE_API
    
    sampler = get_extraction_sampler()
    if sampler is None:
        print("❌ No extraction sampler available. Set EXTRACTION_ENDPOINT or PROCESSOR=gpt-4o-mini")
        return
    
    print(f"✓ Extraction sampler initialized: {sampler.model} at {sampler.client.base_url}")
    print(f"\nTesting {len(samples)} GPQA samples...\n")
    
    correct = 0
    for i, sample in enumerate(samples, 1):
        doc = sample['doc']
        target = sample['target']
        response = sample['resps'][0][0]  # First response
        
        # Build question with choices for extraction context
        question = QUERY_TEMPLATE_API.format(
            Question=doc["Question"],
            choice1=doc["choice1"],
            choice2=doc["choice2"],
            choice3=doc["choice3"],
            choice4=doc["choice4"]
        )
        
        # Extract answer
        extracted = extract_answer(sampler, question, response)
        
        # Normalize
        if extracted in ["a", "b", "c", "d"]:
            extracted = extracted.upper()
        if extracted not in ["A", "B", "C", "D"]:
            extracted = "A"  # Default fallback
        
        is_correct = (extracted == target)
        correct += is_correct
        
        status = "✓" if is_correct else "✗"
        print(f"{status} Sample {i}: Extracted={extracted}, Target={target}")
        if not is_correct:
            print(f"  Response preview: {response[:200]}...")
            print(f"  Extracted text: '{extracted}'")
    
    print(f"\n{'='*60}")
    print(f"Accuracy: {correct}/{len(samples)} = {100*correct/len(samples):.1f}%")
    print(f"{'='*60}")


def test_math_extraction(samples):
    """Test MATH extraction."""
    from lm_eval.tasks.openai_math.utils import extract_answer_idx, EXTRACTION_TEMPLATE_IDX
    
    sampler = get_extraction_sampler()
    if sampler is None:
        print("❌ No extraction sampler available. Set EXTRACTION_ENDPOINT or PROCESSOR=gpt-4o-mini")
        return
    
    print(f"✓ Extraction sampler initialized: {sampler.model} at {sampler.client.base_url}")
    print(f"\nTesting {len(samples)} MATH samples...\n")
    
    correct = 0
    for i, sample in enumerate(samples[:10], 1):  # Test first 10 samples
        doc = sample['doc']
        target = str(doc['answer'])
        response = sample['resps'][0][0]  # First response
        
        # For testing, use target as the first option
        options = [target]
        options_str = "[" + ", ".join(["'" + str(o) + "'" for o in options]) + "]"
        
        # Extract answer
        idx_str = extract_answer_idx(sampler, options_str, response)
        
        # Process extraction
        if idx_str == "-1" or not idx_str.isdigit():
            extracted = response[:50]  # Show preview if extraction failed
            is_correct = False
        else:
            idx = int(idx_str) - 1
            if 0 <= idx < len(options):
                extracted = options[idx]
                is_correct = (extracted == target)
            else:
                extracted = "OUT_OF_BOUNDS"
                is_correct = False
        
        correct += is_correct
        status = "✓" if is_correct else "✗"
        print(f"{status} Sample {i}: Extracted='{extracted}', Target='{target}'")
        if not is_correct:
            print(f"  Index returned: {idx_str}")
            print(f"  Response preview: {response[:150]}...")
    
    print(f"\n{'='*60}")
    print(f"Accuracy: {correct}/{min(10, len(samples))} = {100*correct/min(10, len(samples)):.1f}%")
    print(f"{'='*60}")


def test_aime_extraction(samples):
    """Test AIME extraction."""
    from lm_eval.tasks.aime.utils import extract_answer_idx, EXTRACTION_TEMPLATE_IDX
    
    sampler = get_extraction_sampler()
    if sampler is None:
        print("⚠ No extraction sampler available. AIME can work without it for integer answers.")
        print("  Testing basic integer extraction only...\n")
    else:
        print(f"✓ Extraction sampler initialized: {sampler.model} at {sampler.client.base_url}")
        print(f"\nTesting {len(samples)} AIME samples...\n")
    
    correct = 0
    for i, sample in enumerate(samples[:10], 1):  # Test first 10 samples
        doc = sample['doc']
        target = str(doc['answer'])
        response = sample['resps'][0][0]  # First response
        
        # Simple digit extraction (AIME answers are 000-999)
        extracted = None
        for word in response.split():
            if word.isdigit():
                extracted = str(int(word))  # Normalize: 023 -> 23
                break
        
        if extracted is None and sampler is not None:
            # Try LLM extraction
            options = [target]
            options_str = "[" + ", ".join(["'" + str(o) + "'" for o in options]) + "]"
            idx_str = extract_answer_idx(sampler, options_str, response)
            
            if idx_str != "-1" and idx_str.isdigit():
                idx = int(idx_str) - 1
                if 0 <= idx < len(options):
                    extracted = options[idx]
        
        if extracted is None:
            extracted = "NOT_FOUND"
        
        is_correct = (extracted == target)
        correct += is_correct
        
        status = "✓" if is_correct else "✗"
        print(f"{status} Sample {i}: Extracted='{extracted}', Target='{target}'")
        if not is_correct:
            print(f"  Response preview: {response[:150]}...")
    
    print(f"\n{'='*60}")
    print(f"Accuracy: {correct}/{min(10, len(samples))} = {100*correct/min(10, len(samples)):.1f}%")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Test extraction on existing sample responses")
    parser.add_argument("--samples", 
                       default="/Users/Anthony/Documents/Uni Work/TUM/Aligning Generative AI with Social Values/Safety-Tax/eval_results/ArliAI__gpt-oss-20b-Derestricted/samples_gpqa_diamond_openai_2026-01-16T17-40-16.614253.jsonl",
                       help="Path to samples JSONL file")
    parser.add_argument("--task", default="gpqa", choices=["gpqa", "math", "aime"], 
                       help="Task type (gpqa, math, or aime)")
    parser.add_argument("--limit", type=int, help="Limit number of samples to test")
    
    args = parser.parse_args()
    
    # Load samples
    print(f"Loading samples from: {args.samples}")
    samples = load_samples(args.samples)
    
    if args.limit:
        samples = samples[:args.limit]
    
    print(f"Loaded {len(samples)} samples\n")
    
    # Check if extraction endpoint is set
    if not os.getenv("EXTRACTION_ENDPOINT") and not os.getenv("PROCESSOR"):
        print("⚠ Warning: Neither EXTRACTION_ENDPOINT nor PROCESSOR environment variable is set")
        print("  Set EXTRACTION_ENDPOINT=<your-modal-url> to test Modal extraction")
        print("  or PROCESSOR=gpt-4o-mini with OPENAI_API_KEY for OpenAI extraction\n")
    
    # Run appropriate test
    if args.task == "gpqa":
        test_gpqa_extraction(samples)
    elif args.task == "math":
        test_math_extraction(samples)
    elif args.task == "aime":
        test_aime_extraction(samples)


if __name__ == "__main__":
    main()
