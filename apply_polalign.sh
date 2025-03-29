#!/bin/bash

# Input arguments
REGION=""
declare -a MS_IN=()
RM_CSV=""

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -region) # input region
            REGION="$2"
            shift 2
            ;;
        -ms) # input MS array (can accept multiple values)
            shift  # skip the -ms flag
            while [[ "$#" -gt 0 ]] && [[ ! "$1" =~ ^- ]]; do
                MS_IN+=("$1")
                shift
            done
            ;;
        -rm_csv) # input CSV with offset and RM
            RM_CSV="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Check if required arguments are provided
if [ -z "$REGION" ]; then
    echo "Error: -region argument is required."
    exit 1
fi

if [ ${#MS_IN[@]} -eq 0 ]; then
    echo "Error: at least one -ms argument is required."
    exit 1
fi

if [ -z "$RM_CSV" ]; then
    echo "Error: -rm_csv argument is required."
    exit 1
fi

# Convert paths to absolute paths
REGION=$(realpath "$REGION")
RM_CSV=$(realpath "$RM_CSV")

# Script directory path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for ms in "${MS_IN[@]}"; do
    RUNFOLDER="polimaging_${ms##*/}"
    MS=$(realpath "$ms")

    mkdir -p "$RUNFOLDER"
    cd "$RUNFOLDER"

    # Imaging
    source "$SCRIPT_DIR/scripts/wsclean_imaging_0.3.sh" "$MS"

    # Run polarisation alignment with measurement set
    python "$SCRIPT_DIR/scripts/polalign.py" \
      --region_file "$REGION" \
      --msin "$MS" \
      --RM_offset_csv "$RM_CSV" \
      --applycal

    cd ../
done

mkdir -p output
mv */polalign*.ms output
mv */*_polrot.h5 output
