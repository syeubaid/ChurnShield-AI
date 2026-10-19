#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  ChurnShield — Startup Script
# ─────────────────────────────────────────────────────────────

export PATH="$HOME/.local/bin:$PATH"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="python3"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║        ChurnShield — Churn Prediction        ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

cd "$PROJECT_DIR"

# Train model if not already trained
if [ ! -f "models/churn_model.pkl" ]; then
  echo "  Training ML model (first run)..."
  $PYTHON model_trainer.py
  echo ""
fi

echo "  Starting Flask server on http://localhost:5000"
echo "  Press Ctrl+C to stop"
echo ""

$PYTHON app.py
