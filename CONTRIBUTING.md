# Contributing

Thanks for helping improve OpenAI Autodata.

## Development setup

```bash
python -m venv .venv
pip install -r requirements.txt
python -m unittest discover -v
```

The tests must run without an OpenAI API key and without network access.

## Pull requests

1. Keep changes focused and explain the behavior they change.
2. Add or update offline tests for logic changes.
3. Run `python -m unittest discover -v` before opening the pull request.
4. Do not commit `.env`, API keys, cost ledgers, generated datasets, PDFs, or extracted third-party paper text.
5. Preserve the fail-closed acceptance behavior: missing rollouts or invalid judge output must never make an item easier to accept.

## Issues

Bug reports should include the command used, Python version, traceback with secrets removed, and expected behavior. Feature requests should explain the intended workflow and how it can be tested without paid API calls where possible.
