#!/bin/bash

# Default values
INPUT_DIR="./"
OUTPUT_DIR="./"

# Initialize variables for required arguments
REGION=""
MS_IN=""
RM_CSV=*/rm_offset_data.csv

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -region)
            REGION="$2"
            shift
            ;;
        -ms)
            MS_IN="$2"
            shift
            ;;
        -rm_csv)
            RM_CSV="$2"
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

# Check if required arguments are provided
if [ -z $REGION ]; then
    echo "Error: -region argument is required."
    exit 1
fi

if [ -z $MS_IN ]; then
    echo "Error: -ms argument is required."
    exit 1
fi


# Script directory path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run the Python script
python "$SCRIPT_DIR/scripts/polalign.py" \
  --input_directory $INPUT_DIR \
  --output_directory $OUTPUT_DIR \
  --region_file $REGION \
  --input_ms $MS_IN \
  --RM_offset_CSV $RM_CSV \
  --applycal
