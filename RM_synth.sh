#!/bin/bash

# Script directory path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run script
python $SCRIPT_DIR/scripts/do_RMsynt.py \
    --input_directory "./" \
    --output_directory "./"
