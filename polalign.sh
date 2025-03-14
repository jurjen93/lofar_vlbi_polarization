#!/bin/bash

#INPUT
H5_IN=$1
REGION=$2

# Script directory path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run script
python $SCRIPT_DIR/scripts/polalign.py \
  --input_directory "./" \
  --output_directory "./" \
  --input_h5 ${H5_IN} \
  --region_file ${REGION}
