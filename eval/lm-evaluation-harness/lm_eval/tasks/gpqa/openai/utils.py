from collections import Counter
import os
import time
from typing import Any, Dict, List, Optional
import random
import re

import datasets

from lm_eval.tasks._extraction_utils import (
    get_extraction_sampler,
    last_boxed_only_string,
    remove_boxed,
    ANSWER_PATTERN,
)

QUERY_TEMPLATE = "{Question}\n\nA) {choice1}\nB) {choice2}\nC) {choice3}\nD) {choice4}"
QUERY_TEMPLATE_API = "{Question}\nAnswer Choices:\n(A) {choice1}\n(B) {choice2}\n(C) {choice3}\n(D) {choice4}"

if os.getenv("PROMPTLONG") is not None:
    QUERY_TEMPLATE += '\n\nAnswer after a long amount of thinking. If you feel like you are finished early, spend the extra time trying to double-check your work until you are absolutely sure that you have the correct answer.'
elif os.getenv("PROMPTSHORT") is not None:
    QUERY_TEMPLATE += '\n\nAnswer after a short amount of thinking. Do not spend excessive time double-checking your work.'
elif os.getenv("PROMPTTOKEN") is not None:
    QUERY_TEMPLATE += f'\n\nThink for up to ' + os.getenv("PROMPTTOKEN") + ' tokens.'
elif os.getenv("PROMPTSTEP") is not None:
    QUERY_TEMPLATE += f'\n\nThink for up to ' + os.getenv("PROMPTSTEP") + ' steps.'

print("QUERY_TEMPLATE: ", QUERY_TEMPLATE)

# ANSWER_PATTERN moved to _extraction_utils.py

EXTRACTION_TEMPLATE = r"""
Look at the following question and an attempt by a student and extract which choice among A, B, C, D the student picked. If the student did not pick any choice, respond with "-1".

IMPORTANT: The student's FINAL answer is what matters. Look especially at:
- The END of the response for conclusions
- Phrases like "So answer X", "Thus X", "Therefore X", "Hence X"
- "The answer is X" or "correct answer is X"
- Boxed answers like \boxed{X}
- If the student repeats their answer multiple times, that IS their answer

Examples:

    Question: ...
    Attempt: Answer: **A**

A

    Question: A) Dinosaur B) Elephant C) Cat D) Dog
    Attempt: ...The answer is therefore Elephant...

B

    Question: ...
    Attempt: Answer: None of the above

-1

    Question: ...
    Attempt: ...Answer: D), because...

D

    Question: ...
(A) 7 
(B) 8 
(C) 4 
(D) 10
    Attempt: 4

C

    Question: ...
    Attempt: ...\\boxed{C}...

C

    Question: ...
    Attempt: ...calculations...So answer D. ...more text...So answer D.

D

    Question: ...
    Attempt: ...Thus the correct choice is **B) some text**...

B

    Question: ...
    Attempt: ...Hence, C is correct...

C

    Question: ...
    Attempt: ...Therefore A...Therefore A...Therefore A...

A

---

YOUR TASK

Read the ENTIRE attempt carefully, paying special attention to the FINAL conclusion. Respond only with the capitalized alphabetic letter (without quotes) or -1. Do not include a rationale.

    Question: %(expression1)s
    Attempt: %(expression2)s
""".strip()

def extract_answer(sampler, question: str, attempt: str):
   prompt = EXTRACTION_TEMPLATE % {"expression1": question, "expression2": attempt}
   response = sampler([dict(content=prompt, role="user")])
   return response

# ChatCompletionSampler moved to _extraction_utils.py

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
    if sampler is not None:
        question = QUERY_TEMPLATE_API.format(Question=doc["Question"], choice1=doc["choice1"], choice2=doc["choice2"], choice3=doc["choice3"], choice4=doc["choice4"])
    else:
        print("No extraction sampler configured. Set EXTRACTION_ENDPOINT or PROCESSOR=gpt-4o-mini for best results.")
        question = None

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

        if a in ["a", "b", "c", "d"]:
            a = a.upper()

        if a not in ["A", "B", "C", "D"]:
            if sampler is not None:
                extracted = extract_answer(sampler, question, a)
                if extracted is not None:
                    a = extracted.strip()
            else:
                pass # TODO: Maybe add back legacy processing

        if a not in ["A", "B", "C", "D"]:
            # print(f"Warning: Default to A as given {results[i-1]} extracted {a}")
            a = "A"

        metrics["extracted_answers"].append(a)
        a = int(a == doc["answer"])
        # if not(a): # Optional logging
        #     print("Marked incorrect\na " + metrics["extracted_answers"][-1] + "\ndoc['answer'] " + doc["answer"])
        if i == 1:
            metrics["exact_match"] = a
            if "exact_matches" in metrics:
                metrics["exact_matches"].append(a)
        elif i > 1:
            metrics["exact_matches"].append(a)
            if i in n_res_list:
                metrics[f"cov@{i}"] = int(1 in metrics["exact_matches"])
                most_common = Counter(metrics["extracted_answers"]).most_common(1)
                metrics[f"maj@{i}"] = int(doc["answer"] == most_common[0][0]) if most_common else 0

    return metrics

# last_boxed_only_string and remove_boxed moved to _extraction_utils.py

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        choices = [
            doc["Incorrect Answer 1"],
            doc["Incorrect Answer 2"],
            doc["Incorrect Answer 3"],
            doc["Correct Answer"],
        ]

        random.shuffle(choices)
        correct_answer_index = choices.index(doc["Correct Answer"])

        out_doc = {
            "choice1": choices[0],
            "choice2": choices[1],
            "choice3": choices[2],
            "choice4": choices[3],
            "answer": f"{chr(65 + correct_answer_index)}",
        }
        return out_doc

    return dataset.map(_process_doc)

def doc_to_text_gpqa(doc: dict) -> str:
    return QUERY_TEMPLATE.format(Question=doc["Question"], choice1=doc["choice1"], choice2=doc["choice2"], choice3=doc["choice3"], choice4=doc["choice4"])