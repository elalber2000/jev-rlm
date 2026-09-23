import json
import os
import random
import time
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

from rlm.rlm_repl import RLM_REPL


use_jev: bool = False
RANDOM_SEED = 42
DATA_DIR = Path(__file__).parent / "data"
LOG_PATH = DATA_DIR / "logs.txt"
RESULTS_PATH = DATA_DIR / "results.json"


def log_progress(message: str) -> None:
    """Append and flush a concise progress line while a run is in progress."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as progress_file:
        progress_file.write(f"[progress] {message}\n")
        progress_file.flush()


@contextmanager
def config_logs():
    """Append this run's stdout log to data/logs.txt."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8", buffering=1) as log_file, redirect_stdout(log_file):
        yield


def generate_massive_context(num_lines: int = 1_000_000, answer: str = "1298418") -> str:
    print(f"Generating context with {num_lines:,} lines...")
    log_progress(f"Generating context: 0/{num_lines:,} lines")

    random_words = ["blah", "random", "text", "data", "content", "information", "sample"]
    lines = []
    progress_interval = max(1, num_lines // 10)
    for index in range(num_lines):
        num_words = random.randint(3, 8)
        line_words = [random.choice(random_words) for _ in range(num_words)]
        lines.append(" ".join(line_words))
        if (index + 1) % progress_interval == 0:
            log_progress(f"Generating context: {index + 1:,}/{num_lines:,} lines")

    lower_position = min(400_000, max(0, num_lines * 2 // 5))
    upper_position = min(num_lines - 1, max(lower_position, num_lines * 3 // 5))
    magic_position = random.randint(lower_position, upper_position)
    lines[magic_position] = f"The magic number is {answer}"
    print(f"Magic number inserted at position {magic_position}")
    return "\n".join(lines)


def _request_chars(value) -> int:
    """Count the serialized characters sent in a request."""
    return len(json.dumps(value, ensure_ascii=False, default=str))


def _append_result(entry: dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH, "r", encoding="utf-8") as results_file:
                results = json.load(results_file)
        except json.JSONDecodeError:
            results = []
    else:
        results = []
    if not isinstance(results, list):
        raise ValueError(f"{RESULTS_PATH} must contain a JSON list")
    results.append(entry)
    with open(RESULTS_PATH, "w", encoding="utf-8") as results_file:
        json.dump(results, results_file, indent=2, ensure_ascii=False)
        results_file.write("\n")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    log_progress(f"Run starting; random seed={RANDOM_SEED}")
    random.seed(RANDOM_SEED)
    answer = str(random.randint(1000000, 9999999))
    context = generate_massive_context(num_lines=100_000, answer=answer)
    log_progress(f"Context ready: {len(context):,} characters; expected answer={answer}")

    log_progress("Initializing RLM and Jev clients")
    rlm = RLM_REPL(
        model="deepseek/deepseek-v4-flash-0731",
        recursive_model="deepseek/deepseek-v4-flash-0731",
        jev_model="typesafe/jev-1.13" if use_jev else None,
        provider="openrouter",
        enable_logging=True,
        max_iterations=10,
        progress_callback=log_progress,
    )
    query = "I'm looking for a magic number. What is it?"

    start = time.perf_counter()
    log_progress("Starting completion")
    with config_logs():
        print(f"Run started; use_jev={use_jev}; seed={RANDOM_SEED}")
        result = rlm.completion(context=context, query=query)
        print(f"Result: {result}. Expected: {answer}")
    elapsed = time.perf_counter() - start
    log_progress(f"Completion returned after {elapsed:.1f}s; collecting call counts")

    root_llm_events = [event for event in rlm.trace_events if event.get("type") == "root_llm"]
    recursive_llm_events = [event for event in rlm.trace_events if event.get("type") == "recursive_llm"]
    jev_events = [event for event in rlm.trace_events if event.get("type") == "jev_request"]
    _append_result({
        "use_jev": use_jev,
        "seed": RANDOM_SEED,
        "num_llm_calls": len(root_llm_events),
        "num_recursive_llm_calls": len(recursive_llm_events),
        "num_jev_calls": len(jev_events),
        "total_llm_chars_in": sum(_request_chars(event.get("request", "")) for event in root_llm_events),
        "total_recursive_llm_chars_in": sum(_request_chars(event.get("request", "")) for event in recursive_llm_events),
        "total_jev_chars_in": sum(_request_chars(event.get("request", "")) for event in jev_events),
        "time": elapsed,
        "result": result,
    })
    log_progress("Run summary saved to data/results.json")


if __name__ == "__main__":
    main()
