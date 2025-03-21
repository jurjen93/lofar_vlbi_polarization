#!/bin/bash

#INPUT
REGION=$1
MS_IN=$2
REF_RM=$3
REF_OFFSET=$4

# Script directory path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run script
python $SCRIPT_DIR/scripts/polalign.py \
  --input_directory "./" \
  --output_directory "./" \
  --region_file ${REGION} \
  --input_ms ${MS_IN} \
  --ref_RM ${REF_RM} \
  --ref_offset ${REF_OFFSET}
