#!/usr/bin/env bash
set -e

# Activate virtual environment
source .venv/bin/activate

# Run the Gradio application
python gradio_app.py "$@"

