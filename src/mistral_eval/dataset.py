import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from datasets import load_dataset


CHOICE_LETTERS = ["A", "B", "C", "D"]

def format_mmlu_prompt(question: str, choices: List[str]) -> str:
    """
    Format a multiple-choice question and its choices into a prompt string
    """

    options = "\n".join(f"{CHOICE_LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return (
        "Read the following multiple-choice question and respond with a single letter"
        "among A, B, C, or D corresponding to the correct answer.\n\n"
        f"Question: {question}\n{options}\n\nAnswer:"
    )


def build_mmlu_sample(
    subjects: List[str] = ["college_mathematics", "high_school_statistics", "machine_learning"],
    n_per_subject: int = 100,
    seed: int = 42,
    output_path: str = "data/sample_eval.jsonl",
) -> List[Dict]:
    """
    Build a sample evaluation dataset from the MMLU benchmark
    """

    rng = random.Random(seed)
    records = []
    for subj in subjects:
        ds = load_dataset("cais/mmlu", subj, split="test")
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        for idx in indices[:n_per_subject]:
            row = ds[idx]
            target_letter = CHOICE_LETTERS[int(row["answer"])]
            records.append({
                "id": f"{subj}_{idx}",
                "subject": subj,
                "prompt": format_mmlu_prompt(row["question"], row["choices"]),
                "target": target_letter,
            })
    rng.shuffle(records)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return records


def split_calibration_test(
    records: List[Dict], cal_ratio: float = 0.5, seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split the records into calibration and test sets based on the specified ratio
    """
    
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n_cal = int(len(shuffled) * cal_ratio)
    return shuffled[:n_cal], shuffled[n_cal:]
