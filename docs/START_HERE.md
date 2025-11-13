# 🚀 Start Here - FPL Bot Platform

## Quick Navigation

### 📖 Documentation
- **`README.md`** - Complete platform guide, features, and usage
- **`examples/README.md`** - Bot development guide with chip usage

### 🤖 Example Bots
Located in `examples/`:
- **`example_platform_bot.py`** - Advanced bot with strategic chip timing
- **`simple_example_bot.py`** - Beginner-friendly template (start here!)
- **`test_example_bots.py`** - Test your bot before submitting

## Getting Started (3 Steps)

### 1️⃣ Launch Platform
```bash
python run_platform.py
# Visit http://127.0.0.1:5001
```

### 2️⃣ Try Example Bots
```bash
cd examples
python test_example_bots.py
# Both bots should pass all tests ✓
```

### 3️⃣ Build Your Bot
```bash
# Copy the simple bot as your template
copy examples\simple_example_bot.py my_bot.py

# Edit my_bot.py to customize chip timing and logic
# Test it: python examples\test_example_bots.py

# Submit via platform UI:
# Competition → Submit Bot → Upload my_bot.py
```

## Bot Competition Features

✅ **Full Chip Support** - Use all 4 FPL chips strategically
✅ **Real Data** - Backtest on historical seasons or current season
✅ **Fair Competition** - Temporal filtering ensures no data leakage
✅ **Example Bots** - Production-ready templates included

## Chip Support

All 4 FPL chips work in bot competition:

| Chip | Use It With | Effect |
|------|-------------|--------|
| Triple Captain | `'tc'` or `'triple_captain'` | Captain gets 3× points |
| Bench Boost | `'bb'` or `'bench_boost'` | Bench players score |
| Wildcard | `'wildcard'` or `'wildcard_1'` | Unlimited transfers |
| Free Hit | `'fh'` or `'free_hit'` | Temporary team |

**Example:**
```python
def optimize(self, data):
    if data['gameweek'] == 5 and 'triple_captain' in data['chips_available']:
        return {
            'captain': best_player_id,
            'vice_captain': second_best_id,
            'lineup': lineup_ids,
            'chip': 'triple_captain'  # Use the chip!
        }
```

## Help & Resources

- 📖 **Full Documentation**: Read `README.md`
- 🤖 **Bot Guide**: Read `examples/README.md`
- 💻 **Platform UI**: Competition → Documentation
- 🧪 **Test Bots**: `python examples/test_example_bots.py`

## What's New

### ✨ Chip Support Added
- All 4 FPL chips fully implemented
- Strategic chip timing in example bots
- Complete documentation with examples

### 📁 Organized Structure
- Example bots in dedicated directory
- Consolidated documentation (no scattered files)
- Clear path from docs to bot submission

## Quick Reference

```bash
# Test examples
cd examples && python test_example_bots.py

# Copy template
copy examples\simple_example_bot.py my_awesome_bot.py

# Launch platform
python run_platform.py

# Submit bot
# Go to: http://127.0.0.1:5001/competition/submit
```

---

**Ready?** Start with `README.md` for the full platform guide, or jump straight to `examples/` to see working bots with chip support! 🎉

