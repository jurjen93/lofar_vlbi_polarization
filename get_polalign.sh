#!/bin/bash

#INPUT
REGION=$1

# Script directory path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run script
python $SCRIPT_DIR/scripts/polalign.py \
  --input_directory "./" \
  --output_directory "./" \
  --region_file ${REGION}
