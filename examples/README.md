# Example Bots

This directory contains sample bots and helper scripts that demonstrate how to build
and validate entries for the FPL Vibes competition.

## Files

| File | Description |
| --- | --- |
| `simple_example_bot.py` | Beginner friendly template that highlights the required methods and a basic chip strategy. |
| `example_platform_bot.py` | A richer reference implementation that showcases more advanced decision making. |
| `test_example_bots.py` | Utility script that validates the example bots to ensure they conform to the platform interface. |

## Getting Started

```bash
# Duplicate the simple bot and make it your own
copy examples\simple_example_bot.py my_bot.py

# Test your bot locally
python tools\bot_tester.py my_bot.py
```

See `docs/BOT_DEVELOPMENT_TOOLS.md` for a deeper walkthrough of the local tooling.


