"""Print SFT dataset stats and check the success criteria."""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import tiktoken

REPO_ROOT = Path(__file__).resolve().parent.parent
PATH = REPO_ROOT / "training" / "sft_data" / "sft_warmup.jsonl"


def main():
    enc = tiktoken.get_encoding("cl100k_base")
    n = 0
    completion_token_lens: list[int] = []
    prompt_token_lens: list[int] = []
    decisions: collections.Counter = collections.Counter()

    with PATH.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            n += 1
            completion_token_lens.append(len(enc.encode(row["completion"])))
            prompt_token_lens.append(len(enc.encode(row["prompt"])))
            try:
                d = json.loads(row["completion"]).get("decision", "")
            except Exception:
                d = "<bad-json>"
            decisions[d] += 1

    mean_c = sum(completion_token_lens) / max(1, n)
    mean_p = sum(prompt_token_lens) / max(1, n)
    shares = {k: v / n for k, v in decisions.items()}
    max_share = max(shares.values()) if shares else 0.0
    classes_present = set(decisions.keys()) & {"approve", "flag", "block", "escalate"}

    pass_n = n >= 200
    pass_len = 30 <= mean_c <= 120
    pass_all4 = len(classes_present) == 4
    pass_no_dom = max_share <= 0.70

    print(f"path: {PATH}")
    print(f"n_examples            : {n}                {'PASS' if pass_n else 'FAIL'} (>=200)")
    print(f"mean_completion_tokens: {mean_c:.1f}            {'PASS' if pass_len else 'FAIL'} (30-120)")
    print(f"mean_prompt_tokens    : {mean_p:.1f}")
    print(f"decision_counts       : {dict(decisions)}")
    print(f"decision_shares       : {{ {', '.join(f'{k}: {v:.3f}' for k, v in shares.items())} }}")
    print(f"all_4_classes         : {sorted(classes_present)}     {'PASS' if pass_all4 else 'FAIL'}")
    print(f"max_class_share       : {max_share:.3f}             {'PASS' if pass_no_dom else 'FAIL'} (<=0.70)")
    overall = "PASS" if (pass_n and pass_len and pass_all4 and pass_no_dom) else "FAIL"
    print(f"overall               : {overall}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
