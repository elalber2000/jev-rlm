"""Prompt templates for the RLM REPL client."""

from typing import Dict

DEFAULT_QUERY = "Please read through the context and answer any queries or respond to any instructions contained within it."

REPL_SYSTEM_PROMPT = """Answer the user's query using `context` and the REPL. Write Python actions in ```repl code blocks. Available tools:
- `llm_query(prompt: str) -> str`: ask the recursive language model to analyze text.
- When enabled: `noul(state: object, instructions: str, true_criteria: str, false_criteria: str) -> dict`, `choice(state: object, instructions: str, criteria: dict[str, str]) -> dict`, `score(state: object, instructions: str, levels: list[str]) -> dict`.

Call these functions directly by name. Do not probe for them with `globals()`, `locals()`, or `builtins`; those introspection functions are disabled. Return the final answer with `FINAL(answer)` or `FINAL_VAR(variable_name)` only after the required work is complete.

Keep work focused. Do not print or copy the entire context into the conversation. Small contexts may be handled directly in the REPL. For large contexts, the workflow below is compulsory."""


def build_system_prompt(jev_enabled: bool = False) -> list[Dict[str, str]]:
    content = REPL_SYSTEM_PROMPT
    if jev_enabled:
        content += "\nJev signatures: `noul(state: object, instructions: str, true_criteria: str, false_criteria: str) -> dict[str, object]` returns a decision; `choice(state: object, instructions: str, criteria: dict[str, str]) -> dict[str, object]` returns a category decision; `score(state: object, instructions: str, levels: list[str]) -> dict[str, object]` returns a rating decision. `state` is the data to judge, `instructions` says what to assess, and criteria/levels define outcomes. In this run, `noul` returned `{'type': str, 'noul': float}`; interpret the numeric `noul` score and keep high-scoring chunks for `llm_query(prompt: str) -> str` evidence extraction.\n"
    else:
        content += "\nJev functions are unavailable. For large contexts, use `llm_query(prompt: str) -> str` once on EVERY chunk. Do this in one REPL code block: loop over all chunks, save each `(chunk_id, response)`, and finish the loop before interpreting results. Do not query only chunk 0 or wait for extra root turns to process chunks one at a time."

    content += """

MANDATORY WORKFLOW FOR LARGE CONTEXTS - COMPLETE BEFORE ANSWERING!!!:
1. For a large text context, split it into manageable, non-overlapping chunks of at most 100,000 characters each. Keep each chunk's ID and source boundaries; avoid smaller chunks unless a tool reports a size limit. Never scan the full context locally instead.
2. With Jev, call `noul` on every chunk and save each `(chunk_id, result)`. For the observed `{'type': str, 'noul': float}` result, use the numeric score to identify likely relevant chunks; do not treat the `type` label as the decision or silently drop results. Do not use `globals()` or `locals()`.
3. Call `llm_query(prompt: str) -> str` on every relevant chunk, or on EVERY chunk when Jev is unavailable. Without Jev, make one REPL loop over all chunks, save each `(chunk_id, response)`, and complete the loop before analyzing results. Verify any answer against its source chunk. Do not claim “not found” unless every chunk was processed and every response checked. Fix workflow errors before answering.
"""
    return [{"role": "system", "content": content}]


USER_PROMPT = 'Answer the original query: "{query}". Continue the required context workflow in the REPL and return the next action.'


def next_action_prompt(query: str, iteration: int = 0, final_answer: bool = False) -> Dict[str, str]:
    if final_answer:
        return {
            "role": "user",
            "content": "If the required context workflow is complete, return the answer using FINAL or FINAL_VAR. Otherwise continue the workflow.",
        }
    if iteration == 0:
        content = "Inspect the context's type and size, then follow the system prompt's mandatory workflow. Do not answer yet.\n\n"
    else:
        content = "Continue from your previous REPL work.\n\n"
    return {"role": "user", "content": content + USER_PROMPT.format(query=query)}
