#!/bin/bash

# Default values
INPUT_DIR="./"
OUTPUT_DIR="./"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -region)
            REGION_FILE="$2"
            shift
            ;;
        -in_dir)
            INPUT_DIR="$2"
            shift
            ;;
        -out_dir)
            OUTPUT_DIR="$2"
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
    shift
done

# Check if required argument is provided
if [ -z $REGION_FILE ]; then
    echo "Error: --region_file argument is required."
    exit 1
fi

# Script directory path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run script
python "$SCRIPT_DIR/scripts/polalign.py" \
  --input_directory $INPUT_DIR \
  --output_directory $OUTPUT_DIR \
  --region_file $REGION_FILE