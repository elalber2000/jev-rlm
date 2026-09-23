# Recursive Language Models (minimal version) 

[Link to the official RLM codebase](https://github.com/alexzhang13/rlm)

[Link to the paper](https://arxiv.org/abs/2512.24601v1)

[Link to the original blogpost 📝](https://alexzhang13.github.io/blog/2025/rlm/)

This is a small experiment to test the RLM architecture with the new Jev model. Run it with `uv run python main.py` after placing the input text in `data/context.txt`. Set `USE_JEV` in `main.py` to `True` or `False` to compare runs. Results are appended to `data/reslts.json`, with interaction traces in `data/traces.jsonl`.
