#!/bin/bash

# Get the absolute path of the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run Python script with paths relative to this script's location
python "$SCRIPT_DIR/scripts/do_RMsynt.py" \
    --input_directory "./" \
    --output_directory "./"
