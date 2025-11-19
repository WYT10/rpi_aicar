#!/usr/bin/env bash
# Simple launcher for the AI Car server

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$ROOT_DIR"

# Activate virtualenv
if [ ! -d "venv" ]; then
  echo "venv/ not found. Create it first with: python3 -m venv venv"
  exit 1
fi

# shellcheck disable=SC1091
source venv/bin/activate

# Run the app
exec python app.py
