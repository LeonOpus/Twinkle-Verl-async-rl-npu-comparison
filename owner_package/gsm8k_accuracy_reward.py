import re


def _extract_last_boxed(text: str) -> str:
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return ""
    start = idx + len("\\boxed{")
    depth = 1
    cursor = start
    while cursor < len(text) and depth > 0:
        if text[cursor] == "{":
            depth += 1
        elif text[cursor] == "}":
            depth -= 1
        cursor += 1
    if depth == 0:
        return text[start : cursor - 1].strip()
    return ""


def _extract_answer(completion: str) -> str:
    text = completion[-500:] if len(completion) > 500 else completion
    boxed = _extract_last_boxed(text)
    if boxed:
        return boxed.replace(",", "").replace(" ", "").strip()
    matches = re.findall(r"####\s*([\-\d,\.\s]+)", text)
    if matches:
        return matches[-1].replace(",", "").replace(" ", "").strip()
    return ""


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> float:
    del data_source, extra_info, kwargs
    predicted = _extract_answer(solution_str)
    if not predicted or not ground_truth:
        return 0.0
    try:
        correct = abs(float(predicted) - float(ground_truth)) < 1e-5
    except (ValueError, OverflowError):
        correct = predicted == ground_truth
    return 1.0 if correct else 0.0
