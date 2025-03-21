#!/bin/bash

# Initialize variables for required arguments
REGION=""
MS_IN=""
RM_CSV=""

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -region) # input region file
            REGION="$2"
            shift
            ;;
        -ms) # array of input MS
            MS_IN+=("$2")
            shift
            ;;
        -rm_csv) # input RM CSV (output from reference observation)
            RM_CSV="$2"
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

if [ ${#MS_IN[@]} -eq 0 ]; then
    echo "Error: at least one -ms argument is required."
    exit 1
fi

if [ -z $RM_CSV ]; then
    echo "Error: -rm_csv argument is required."
    exit 1
fi

$RM_CSV=$(realpath $RM_CSV)
$REGION=$(realpath $REGION)


# Script directory path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for ms in "${MS_IN[@]}"; do

  RUNFOLDER=polimaging_${ms##*/}
  MS=$(realpath $ms)

  mkdir -p $RUNFOLDER
  cd $RUNFOLDER

  source $SCRIPT_DIR/scripts/wsclean_imaging.sh ${MS}

  # Run the Python script
  python $SCRIPT_DIR/scripts/polalign.py \
    --region_file $REGION \
    --msin $MS_IN $MS \
    --RM_offset_csv $RM_CSV \
    --applycal

  cd ../

done

mkdir -p output
mv */polalign*.ms output
mv */*_polrot.h5 output
