#!/bin/bash

# Get the directory of the current script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run Python script with paths relative to the script's location
python "$SCRIPT_DIR/../scripts/do_RMsynth.py" \
    --input_directory "$SCRIPT_DIR" \
    --output_directory "$SCRIPT_DIR"
