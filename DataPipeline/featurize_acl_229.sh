#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-$SCRIPT_DIR/../pipeline_config.json}"

echo "Running config-driven extraction with $CONFIG_PATH"
python3 "$SCRIPT_DIR/../run_combined_extraction.py" "$CONFIG_PATH"

