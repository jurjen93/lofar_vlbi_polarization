#!/bin/bash

# Input
MSIN=$1

# Script directory path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run imaging
source $SCRIPT_DIR/scripts/wsclean_imaging.sh ${MSIN}

# Run RMsynt
python $SCRIPT_DIR/scripts/RMsynt.py \
    --input_directory "./" \
    --output_directory "./"

echo "RM synthesis finished, check --> RMsynthFDF_*.fits"
