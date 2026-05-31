#!/bin/bash

# Input
MSIN=$(realpath $1)

# Script directory path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Make RUNDIR
RUNDIR=RMsynth_ref_output
mkdir -p ${RUNDIR}
cd ${RUNDIR}

# Run imaging
source $SCRIPT_DIR/scripts/imaging_03.sh ${MSIN}
source $SCRIPT_DIR/scripts/imaging_06.sh ${MSIN}
source $SCRIPT_DIR/scripts/imaging_12.sh ${MSIN}

# Run RMsynt
python $SCRIPT_DIR/scripts/RMsynt.py \
    --input_directory "./" \
    --output_directory "./"

echo "RM synthesis finished, check --> RMsynthFDF_*.fits"
