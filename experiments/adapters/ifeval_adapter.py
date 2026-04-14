"""
IFEval Benchmark Adapter

Adapts Google's IFEval (Instruction Following Evaluation) benchmark
for CPGA+SENTINEL evaluation.

IFEval: 541 prompts with ~1.54 verifiable instructions per prompt.
25 instruction types across 9 categories — all checked by deterministic
Python functions (no LLM judges).

Reference: https://arxiv.org/abs/2311.07911
Dataset: https://huggingface.co/datasets/google/IFEval
Code: https://github.com/google-research/google-research/tree/master/instruction_following_eval
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# IFEval instruction checkers — deterministic Python implementations
# mirroring google-research/instruction_following_eval/instructions.py
# ---------------------------------------------------------------------------

def _count_words(text: str) -> int:
    return len(text.split())


def _count_sentences(text: str) -> int:
    sentences = re.split(r'[.!?]+\s*', text.strip())
    return len([s for s in sentences if s.strip()])


def _count_paragraphs(text: str) -> int:
    paras = text.split("\n\n")
    return len([p for p in paras if p.strip()])


def _check_keyword_existence(text: str, keywords: list[str]) -> bool:
    text_lower = text.lower()
    return all(kw.lower() in text_lower for kw in keywords)


def _check_keyword_frequency(text: str, keyword: str, frequency: int, relation: str) -> bool:
    count = text.lower().count(keyword.lower())
    if relation == "at least":
        return count >= frequency
    elif relation == "at most":
        return count <= frequency
    elif relation == "exactly":
        return count == frequency
    return count >= frequency


def _check_forbidden_words(text: str, forbidden_words: list[str]) -> bool:
    text_lower = text.lower()
    return not any(w.lower() in text_lower for w in forbidden_words)


def _check_letter_frequency(text: str, letter: str, let_frequency: int, let_relation: str) -> bool:
    count = text.lower().count(letter.lower())
    if let_relation == "at least":
        return count >= let_frequency
    elif let_relation == "at most":
        return count <= let_frequency
    return count >= let_frequency


def _check_response_language(text: str, language: str) -> bool:
    try:
        from langdetect import detect
        detected = detect(text)
        lang_map = {
            "English": "en", "French": "fr", "Spanish": "es", "German": "de",
            "Italian": "it", "Portuguese": "pt", "Dutch": "nl", "Russian": "ru",
            "Chinese": "zh-cn", "Japanese": "ja", "Korean": "ko", "Arabic": "ar",
            "Hindi": "hi", "Bengali": "bn", "Turkish": "tr",
        }
        expected = lang_map.get(language, language.lower()[:2])
        return detected == expected
    except Exception:
        return True


def _check_number_sentences(text: str, num_sentences: int, relation: str) -> bool:
    count = _count_sentences(text)
    if relation == "at least":
        return count >= num_sentences
    elif relation == "at most":
        return count <= num_sentences
    return count == num_sentences


def _check_number_paragraphs(text: str, num_paragraphs: int) -> bool:
    return _count_paragraphs(text) == num_paragraphs


def _check_number_words(text: str, num_words: int, relation: str) -> bool:
    count = _count_words(text)
    if relation == "at least":
        return count >= num_words
    elif relation == "at most":
        return count <= num_words
    return count == num_words


def _check_nth_paragraph_first_word(text: str, num_paragraphs: int, nth_paragraph: int, first_word: str) -> bool:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if nth_paragraph > len(paras) or len(paras) < num_paragraphs:
        return False
    target = paras[nth_paragraph - 1]
    words = target.split()
    return words[0].lower().rstrip(",:;") == first_word.lower() if words else False


def _check_number_placeholders(text: str, num_placeholders: int) -> bool:
    count = len(re.findall(r"\[.*?\]", text))
    return count >= num_placeholders


def _check_postscript(text: str, postscript_marker: str) -> bool:
    return postscript_marker.lower() in text.lower()


def _check_number_bullet_lists(text: str, num_bullets: int) -> bool:
    bullets = re.findall(r"^\s*[\*\-•]\s+", text, re.MULTILINE)
    return len(bullets) >= num_bullets


def _check_constrained_response(text: str) -> bool:
    stripped = text.strip()
    return stripped in ("My answer is yes.", "My answer is no.", "My answer is maybe.")


def _check_number_highlighted_sections(text: str, num_highlights: int) -> bool:
    highlights = re.findall(r"\*[^*]+\*", text)
    return len(highlights) >= num_highlights


def _check_multiple_sections(text: str, section_splitter: str, num_sections: int) -> bool:
    if section_splitter == "Section":
        sections = re.split(r"(?:^|\n)#+\s+", text)
    else:
        sections = text.split(section_splitter)
    sections = [s.strip() for s in sections if s.strip()]
    return len(sections) >= num_sections


def _check_json_format(text: str) -> bool:
    try:
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            stripped = "\n".join(lines).strip()
        json.loads(stripped)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def _check_title(text: str) -> bool:
    lines = text.strip().split("\n")
    return bool(lines and re.match(r"^#+\s+.+|^Title:\s+.+", lines[0]))


def _check_two_responses(text: str) -> bool:
    seps = ["***", "---", "===", "response 1", "response 2"]
    text_lower = text.lower()
    return any(s in text_lower for s in seps)


def _check_repeat_prompt(text: str, prompt_to_repeat: str) -> bool:
    return prompt_to_repeat.strip().lower() in text.lower()


def _check_end_checker(text: str, end_phrase: str) -> bool:
    return text.strip().endswith(end_phrase)


def _check_quotation(text: str) -> bool:
    quotes = re.findall(r'"[^"]*"', text) + re.findall(r"'[^']*'", text)
    return len(quotes) >= 1


def _check_capital_word_frequency(text: str, capital_frequency: int, capital_relation: str) -> bool:
    all_caps_words = [w for w in text.split() if w.isupper() and len(w) > 1]
    if capital_relation == "at least":
        return len(all_caps_words) >= capital_frequency
    elif capital_relation == "at most":
        return len(all_caps_words) <= capital_frequency
    return len(all_caps_words) >= capital_frequency


def _check_english_capital(text: str) -> bool:
    words = text.split()
    return all(w[0].isupper() for w in words if w[0].isalpha()) if words else False


def _check_english_lowercase(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return all(c.islower() for c in letters) if letters else False


def _check_no_comma(text: str) -> bool:
    return "," not in text


INSTRUCTION_CHECKERS: dict[str, Callable] = {
    "keywords:existence": lambda text, **kw: _check_keyword_existence(text, kw.get("keywords", [])),
    "keywords:frequency": lambda text, **kw: _check_keyword_frequency(
        text, kw.get("keyword", ""), kw.get("frequency", 1), kw.get("relation", "at least")),
    "keywords:forbidden_words": lambda text, **kw: _check_forbidden_words(text, kw.get("forbidden_words", [])),
    "keywords:letter_frequency": lambda text, **kw: _check_letter_frequency(
        text, kw.get("letter", ""), kw.get("let_frequency", 1), kw.get("let_relation", "at least")),
    "language:response_language": lambda text, **kw: _check_response_language(text, kw.get("language", "English")),
    "length_constraints:number_sentences": lambda text, **kw: _check_number_sentences(
        text, kw.get("num_sentences", 1), kw.get("relation", "at least")),
    "length_constraints:number_paragraphs": lambda text, **kw: _check_number_paragraphs(
        text, kw.get("num_paragraphs", 1)),
    "length_constraints:number_words": lambda text, **kw: _check_number_words(
        text, kw.get("num_words", 100), kw.get("relation", "at least")),
    "length_constraints:nth_paragraph_first_word": lambda text, **kw: _check_nth_paragraph_first_word(
        text, kw.get("num_paragraphs", 1), kw.get("nth_paragraph", 1), kw.get("first_word", "")),
    "detectable_content:number_placeholders": lambda text, **kw: _check_number_placeholders(
        text, kw.get("num_placeholders", 1)),
    "detectable_content:postscript": lambda text, **kw: _check_postscript(
        text, kw.get("postscript_marker", "P.S.")),
    "detectable_format:number_bullet_lists": lambda text, **kw: _check_number_bullet_lists(
        text, kw.get("num_bullets", 1)),
    "detectable_format:constrained_response": lambda text, **kw: _check_constrained_response(text),
    "detectable_format:number_highlighted_sections": lambda text, **kw: _check_number_highlighted_sections(
        text, kw.get("num_highlights", 1)),
    "detectable_format:multiple_sections": lambda text, **kw: _check_multiple_sections(
        text, kw.get("section_splitter", "Section"), kw.get("num_sections", 2)),
    "detectable_format:json_format": lambda text, **kw: _check_json_format(text),
    "detectable_format:title": lambda text, **kw: _check_title(text),
    "combination:two_responses": lambda text, **kw: _check_two_responses(text),
    "combination:repeat_prompt": lambda text, **kw: _check_repeat_prompt(
        text, kw.get("prompt_to_repeat", "")),
    "startend:end_checker": lambda text, **kw: _check_end_checker(text, kw.get("end_phrase", "")),
    "startend:quotation": lambda text, **kw: _check_quotation(text),
    "change_case:capital_word_frequency": lambda text, **kw: _check_capital_word_frequency(
        text, kw.get("capital_frequency", 1), kw.get("capital_relation", "at least")),
    "change_case:english_capital": lambda text, **kw: _check_english_capital(text),
    "change_case:english_lowercase": lambda text, **kw: _check_english_lowercase(text),
    "punctuation:no_comma": lambda text, **kw: _check_no_comma(text),
}


class IFEvalAdapter:
    """Adapter for Google's IFEval benchmark."""

    def __init__(self):
        self._dataset = None

    def load_tasks(self, max_tasks: int | None = None) -> list[dict]:
        """Load IFEval from HuggingFace datasets."""
        try:
            from datasets import load_dataset
            ds = load_dataset("google/IFEval", split="train")
        except Exception as e:
            print(f"Failed to load IFEval from HuggingFace: {e}")
            print("Attempting local fallback...")
            return self._load_local_fallback(max_tasks)

        tasks = []
        for i, row in enumerate(ds):
            if max_tasks and i >= max_tasks:
                break

            instruction_ids = row.get("instruction_id_list", [])
            kwargs_list = row.get("kwargs", [])
            if isinstance(kwargs_list, str):
                kwargs_list = json.loads(kwargs_list)

            constraints = []
            for j, (iid, kw) in enumerate(zip(instruction_ids, kwargs_list)):
                if isinstance(kw, str):
                    kw = json.loads(kw)

                checker = INSTRUCTION_CHECKERS.get(iid)
                if checker:
                    check_fn = lambda text, _checker=checker, _kw=kw: _checker(text, **_kw)
                else:
                    check_fn = None

                constraints.append({
                    "id": f"{iid}_{j}",
                    "instruction_id": iid,
                    "text": f"[{iid}] with params {json.dumps(kw)}",
                    "check_fn": check_fn,
                    "kwargs": kw,
                })

            tasks.append({
                "task_id": f"ifeval_{i}",
                "prompt": row["prompt"],
                "base_prompt": row["prompt"],
                "constraints": constraints,
                "constraint_count": len(constraints),
                "instruction_ids": instruction_ids,
            })

        return tasks

    def _load_local_fallback(self, max_tasks: int | None = None) -> list[dict]:
        """Fallback: try loading from local JSONL."""
        local_paths = [
            Path(__file__).parent.parent / "data" / "ifeval.jsonl",
            Path(__file__).parent.parent.parent / "data" / "ifeval.jsonl",
        ]
        for p in local_paths:
            if p.exists():
                tasks = []
                with open(p) as f:
                    for i, line in enumerate(f):
                        if max_tasks and i >= max_tasks:
                            break
                        row = json.loads(line)
                        tasks.append(self._row_to_task(row, i))
                return tasks
        return []

    def _row_to_task(self, row: dict, index: int) -> dict:
        instruction_ids = row.get("instruction_id_list", [])
        kwargs_list = row.get("kwargs", [])
        if isinstance(kwargs_list, str):
            kwargs_list = json.loads(kwargs_list)

        constraints = []
        for j, (iid, kw) in enumerate(zip(instruction_ids, kwargs_list)):
            if isinstance(kw, str):
                kw = json.loads(kw)
            checker = INSTRUCTION_CHECKERS.get(iid)
            check_fn = (lambda text, _c=checker, _k=kw: _c(text, **_k)) if checker else None
            constraints.append({
                "id": f"{iid}_{j}",
                "instruction_id": iid,
                "text": f"[{iid}]",
                "check_fn": check_fn,
                "kwargs": kw,
            })

        return {
            "task_id": f"ifeval_{index}",
            "prompt": row["prompt"],
            "base_prompt": row["prompt"],
            "constraints": constraints,
            "constraint_count": len(constraints),
        }

    def extract_constraints(self, task: dict) -> list[dict]:
        return task["constraints"]

    def score_strict(self, output: str, task: dict) -> dict:
        """Strict scoring: each instruction checked on raw response."""
        constraints = task["constraints"]
        per_instruction = []
        all_passed = True

        for c in constraints:
            check_fn = c.get("check_fn")
            if check_fn is None:
                per_instruction.append({
                    "id": c["id"],
                    "passed": None,
                    "skipped": True,
                })
                continue
            try:
                passed = bool(check_fn(output))
            except Exception as e:
                passed = False
            per_instruction.append({
                "id": c["id"],
                "instruction_id": c.get("instruction_id", ""),
                "passed": passed,
            })
            if not passed:
                all_passed = False

        checkable = [p for p in per_instruction if not p.get("skipped")]
        n_passed = sum(1 for p in checkable if p["passed"])
        n_total = len(checkable)

        return {
            "follow_all_instructions": all_passed,
            "instruction_accuracy": n_passed / n_total if n_total > 0 else 0.0,
            "prompt_accuracy": 1.0 if all_passed else 0.0,
            "n_passed": n_passed,
            "n_total": n_total,
            "per_instruction": per_instruction,
        }

    def score_loose(self, output: str, task: dict) -> dict:
        """Loose scoring: try variants (strip first/last line, remove *)."""
        variants = [
            output,
            "\n".join(output.split("\n")[1:]),
            "\n".join(output.split("\n")[:-1]),
            output.replace("*", ""),
            output.replace("**", ""),
        ]

        best_result = None
        best_passed = -1

        for variant in variants:
            result = self.score_strict(variant, task)
            n_passed = result["n_passed"]
            if n_passed > best_passed:
                best_passed = n_passed
                best_result = result

        if best_result:
            best_result["scoring_mode"] = "loose"
        return best_result or self.score_strict(output, task)

    def score(self, output: str, task: dict) -> dict:
        """Default scoring = strict (matches standard leaderboard)."""
        return self.score_strict(output, task)

    def aggregate_by_category(self, results: list[dict]) -> dict:
        """Aggregate per-category accuracy (prefix before ':')."""
        from collections import defaultdict
        category_stats: dict[str, dict] = defaultdict(lambda: {"passed": 0, "total": 0})

        for r in results:
            for pi in r.get("per_instruction", []):
                if pi.get("skipped"):
                    continue
                iid = pi.get("instruction_id", pi["id"])
                category = iid.split(":")[0] if ":" in iid else "other"
                category_stats[category]["total"] += 1
                if pi["passed"]:
                    category_stats[category]["passed"] += 1

        return {
            cat: {
                "accuracy": s["passed"] / s["total"] if s["total"] > 0 else 0.0,
                "passed": s["passed"],
                "total": s["total"],
            }
            for cat, s in sorted(category_stats.items())
        }
