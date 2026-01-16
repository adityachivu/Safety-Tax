from collections import Counter
import os
import re
import signal
from typing import Dict, List, Optional

import datasets

from lm_eval.utils import eval_logger
from lm_eval.tasks._extraction_utils import (
    get_extraction_sampler,
    last_boxed_only_string,
    remove_boxed,
    ANSWER_PATTERN,
)

if os.getenv("PROMPTSTEP") is not None:
    QUERY_TEMPLATE = '{Question}\n\nThink for up to ' + os.getenv("PROMPTSTEP") + ' steps.'
elif os.getenv("PROMPTTOKEN") is not None:
    QUERY_TEMPLATE = '{Question}\n\nThink for up to ' + os.getenv("PROMPTTOKEN") + ' tokens.'
elif os.getenv("PROMPTLONG") is not None:
    QUERY_TEMPLATE = '{Question}\n\nAnswer after a long amount of thinking. If you feel like you are finished early, spend the extra time trying to double-check your work until you are absolutely sure that you have the correct answer.'
elif os.getenv("PROMPTSHORT") is not None:
    QUERY_TEMPLATE = '{Question}\n\nAnswer after a short amount of thinking. Do not spend excessive time double-checking your work.'
else:
    QUERY_TEMPLATE = '{Question}'

print("QUERY_TEMPLATE: ", QUERY_TEMPLATE)

# ANSWER_PATTERN moved to _extraction_utils.py

EXTRACTION_TEMPLATE_IDX = r"""
Look at the following attempt by a student and extract the student's FINAL answer. If it is equivalent (ignoring trivial simplifications) to any of the provided options, return the index of that option starting from 1. Else, return -1.

IMPORTANT: Focus on the student's FINAL answer. Look especially at:
- The END of the response for final conclusions
- Boxed answers like \boxed{X} - this is almost always the final answer
- Phrases like "So the answer is", "Thus", "Therefore", "Hence"
- If the student repeats an answer multiple times at the end, that IS their answer
- The LAST numerical answer mentioned if multiple are given

Examples:

    Options: ['2x+4', '2x', '4x']
    Attempt: The answer is 3+2x.

-1
(the student's answer is not among the options)

    Options: ['72,000']
    Attempt: 72000 \text{ cents}.

1
(always give benefit of the doubt to units and ignore formatting which makes the 1st option match)

    Options: ['2/(-3)', '2/3']
    Attempt: -1 * 2/3

1
(the 1st option matches after trivial simplifications which are fine)

    Options: ['x=5']
    Attempt: 5

1

    Options: ['\dfrac{33}{100}']
    Attempt: 0.33

1

    Options: ['75^\circ']
    Attempt: ...various calculations and explanations...hence the answer is $\boxed{x in 75}$.

1

    Options: ['(1,-3)', '(1,-1)', '(1,0)', '(1,-2)']
    Attempt: -2, 1

4
(ignore whitespace and other formatting which makes the 4th option match)

    Options: ['-2,1']
    Attempt: 1, -2

1
(likely a problem where multiple solutions are possible thus ignore order)

    Options: ['42']
    Attempt: ...So the answer is 42. ...So the answer is 42. ...So the answer is 42.

1
(repeated answers at the end indicate the final answer)

    Options: ['100', '200', '150']
    Attempt: ...Therefore, the final answer is $\boxed{150}$.

3

    Options: ['12']
    Attempt: ...$\boxed{12^{\mathrm{th}}}$.

1

    Options: ['2516_8']
    Attempt: 2516

1
(give benefit of the doubt for different bases)

    Options: ['11\sqrt2']
    Attempt: 11\sqrt{2}

1

    Options: ['11,\! 111,\! 111,\! 100']
    Attempt: 11111111100

1

    Options: ['\text{Navin}']
    Attempt: ...it is navin.

1

---

YOUR TASK


Respond with only the index of the matching option starting from 1 or -1 if there is absolutely no reasonable match. Do not include a rationale.

    Options: %(expression1)s
    Attempt: %(expression2)s
""".strip()


# https://github.com/openai/simple-evals/blob/580d359553a88584c11ce4efb97d49d9386e0d9e/common.py#L153C1-L156C45
def extract_answer_idx(sampler, options: List[str], attempt: str):
    prompt = EXTRACTION_TEMPLATE_IDX % {"expression1": options, "expression2": attempt}
    response = sampler([dict(content=prompt, role="user")])
    return response

# ChatCompletionSampler moved to _extraction_utils.py

def doc_to_text(doc: dict) -> str:
    return QUERY_TEMPLATE.format(Question=doc["problem"])

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        solution = doc.get("solution", doc.get("orig_solution", doc.get("orig_orig_solution")))
        problem = doc.get("problem", doc.get("orig_problem", doc.get("orig_orig_problem")))
        answer = doc.get("answer", doc.get("orig_answer", doc.get("orig_orig_answer")))
        if solution is None:
            print("Warning: No solution found; DOC:", doc)
        out_doc = {
            "problem": problem,
            "solution": solution,
            "answer": answer,
        }
        if getattr(doc, "few_shot", None) is not None:
            out_doc["few_shot"] = True
        return out_doc
    return dataset.map(_process_doc)

def process_docs_openai_math_cot_quality_check(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        problem = doc.get("orig_problem", doc.get("orig_orig_problem"))
        solution = doc.get("orig_solution", doc.get("orig_orig_solution"))
        answer = doc.get("orig_answer", doc.get("orig_orig_answer"))
        thinking_trajectory = doc.get("thinking_trajectory", doc.get("orig_thinking_trajectory", doc.get("refined_thinking_trajectory")))
        try:
            out_doc = {
                "problem": problem,
                "solution": solution,
                "answer": answer,
                "thinking_trajectory": thinking_trajectory[:-1],
            }
            if getattr(doc, "few_shot", None) is not None:
                out_doc["few_shot"] = True
            return out_doc
        except:
            return {'problem': 'Drop', 'solution': 'Drop', 'answer': 'Drop', 'thinking_trajectory': ['Drop']}
    processed_dataset = dataset.map(_process_doc)
    processed_dataset = processed_dataset.filter(lambda x: x['problem'] != 'Drop')
    return processed_dataset

def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    metrics = {"exact_match": None, "extracted_answers": []}
    # Multiple results -> we are measuring cov/maj etc
    if isinstance(results[0], list):
        results = results[0]
        n_res = len(results) # e.g. 64
        n_res_list = [2**i for i in range(1, int(n_res.bit_length()))] # e.g. [2, 4, 8, 16, 32, 64]
        metrics = {
            **metrics,
            "exact_matches": [],
            **{f"cov@{n}": -1 for n in n_res_list},
            **{f"maj@{n}": -1 for n in n_res_list},
        }

    sampler = get_extraction_sampler()
    if sampler is None:
        print("Warning: No extraction sampler configured. Set EXTRACTION_ENDPOINT or PROCESSOR=gpt-4o-mini")
        raise ValueError("MATH requires extraction sampler. Set EXTRACTION_ENDPOINT=<modal-url> or PROCESSOR=gpt-4o-mini")

    if doc.get("answer") is None:
        print("Warning: No answer found in doc; DOC:", doc)
        gt = ""
    elif isinstance(doc["answer"], str) and doc["answer"].isdigit():
        gt = str(int(doc["answer"])) # 023 -> 23
    else:
        gt = str(doc["answer"])
    split_tokens = ["<|im_start|>answer\n", "<|im_start|>"]

    for i, a in enumerate(results, start=1):
        if split_tokens[0] in a:
            a = a.split(split_tokens[0])[-1]
        elif split_tokens[1] in a:
            a = a.split(split_tokens[1])[-1]
            if "\n" in a:
                a = "\n".join(a.split("\n")[1:])

        if a is None:
            a = ""
        
        if (box := last_boxed_only_string(a)) is not None:
            extracted = remove_boxed(box)
            if extracted is not None:
                a = extracted
        # re.DOTALL is key such that newlines are included e.g. if it does `Answer: Here is the solution:\n\n10`
        elif (matches := re.findall(ANSWER_PATTERN, a, re.DOTALL)) != []:
            a = matches[-1]  # Get the last match

        if a and gt and (a.isdigit()) and (gt.isdigit()):
            a = str(int(a)) # 023 -> 23
        elif sampler is not None:
            options = [gt] + list(set(metrics["extracted_answers"]) - {gt})
            if len(options) > 7:
                # Could switch back to exact returning like in AIME in that case
                # Problem with exact returning is that it sometimes messes up small things like a dollar sign
                print("Warning: Lots of options which may harm indexing performance:", options)
            # This ensures that if doc['answer'] is \text{Evelyn} it is represented as such and not \\text{Evelyn}
            options_str = "[" + ", ".join(["'" + str(o) + "'" for o in options]) + "]"
            idx = extract_answer_idx(sampler, options_str, a)
            if idx is not None and idx != "-1":
                if idx.isdigit():
                    idx = int(idx) - 1
                    if len(options) > idx >= 0:
                        a = options[idx]
                    else:
                        print("Warning: Index out of bounds; leaving answer unchanged\n", a, "\noptions", options_str, "\ndoc['answer']", gt, "\nidx", idx)
                else:
                    print("Warning: Processing did not produce integer index\na", a, "\noptions", options_str, "\ndoc['answer']", gt)
        else:
            pass # TODO: Maybe add back legacy processing

        metrics["extracted_answers"].append(a)
        a = int(a == gt)
        if not(a): # Optional logging
            print("Marked incorrect\na " + metrics["extracted_answers"][-1] + "\ndoc['answer'] " + gt)
        if i == 1:
            metrics["exact_match"] = a
            if "exact_matches" in metrics:
                metrics["exact_matches"].append(a)
        elif i > 1:
            metrics["exact_matches"].append(a)
            if i in n_res_list:
                metrics[f"cov@{i}"] = int(1 in metrics["exact_matches"])
                most_common = Counter(metrics["extracted_answers"]).most_common(1)
                metrics[f"maj@{i}"] = int(gt == most_common[0][0]) if most_common else 0

    return metrics

# last_boxed_only_string and remove_boxed moved to _extraction_utils.py
