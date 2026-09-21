#!/usr/bin/env bash
# Install the pi integration (extension + skill) for the user running this box.
# Idempotent. Extension auto-discovers from ~/.pi/agent/extensions/.
set -e
R="$(cd "$(dirname "$0")/.." && pwd)"   # ocr-bench root
SRC="$R/pi-integration"

mkdir -p ~/.pi/agent/extensions ~/.pi/agent/skills/ocr-bench
cp "$SRC/ocr-bench.ts" ~/.pi/agent/extensions/ocr-bench.ts
cp "$SRC/SKILL.md" ~/.pi/agent/skills/ocr-bench/SKILL.md

# make sure the CLI facade is executable and the venv exists
chmod +x "$R/pi_tools.py" 2>/dev/null || true
if [ ! -x "$R/.venv/bin/python3" ]; then
  echo "note: $R/.venv missing - run $R/setup_env.sh"
fi

echo "installed:"
echo "  ~/.pi/agent/extensions/ocr-bench.ts"
echo "  ~/.pi/agent/skills/ocr-bench/SKILL.md"
echo
echo "smoke test (should print a leaderboard):"
"$R/.venv/bin/python3" "$R/pi_tools.py" leaderboard -n 5 2>/dev/null || \
  python3 "$R/pi_tools.py" leaderboard -n 5
echo
echo "restart pi (or run /reload) to pick up the ocr_* tools."
