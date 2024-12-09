#!/bin/bash
#
# Delete temporary files. Run this script once all files have been verified.
# Usage:
#   ./delete_temp_files.sh <FILE_PARAM>

# Uncomment for full verbose
# set -v

# Immediately exit if error
set -e


# PARAMETERS & VARIABLES
# ==============================================================================

# Retrieve input params
SUBJECT=$1

# Create BIDS architecture
PATH_IN="${PATH_DATA}/${SUBJECT}/anat"
ofolder_seg="${PATH_DATA_PROCESSED}/derivatives/labels/${SUBJECT}/anat"
ofolder_reg="${PATH_DATA_PROCESSED}/${SUBJECT}/anat"

# Set filenames
file_t1w_mts="${SUBJECT}_acq-T1w_MTS"
file_mton="${SUBJECT}_acq-MTon_MTS"
file_mtoff="${SUBJECT}_acq-MToff_MTS"
file_t1w="${SUBJECT}_T1w"
file_t2w="${SUBJECT}_T2w"
file_t2s="${SUBJECT}_T2star"

FILES_SRC=(
  "${file_t1w_mts}_crop_r"
  "${file_mton}_reg"
  "${file_mtoff}_reg"
  "${file_t1w}_reg2"
  "${file_t2w}_reg2"
  "${file_t2s}_reg2"
)

FILES_DEST=(
  "${file_t1w_mts}"
  "${file_mton}"
  "${file_mtoff}"
  "${file_t1w}"
  "${file_t2w}"
  "${file_t2s}"
)



# FUNCTIONS
# ==============================================================================

# Check if an item is contained in at least one element of a list. If so, return 1.
# Usage: contains LIST STR
# Source: https://stackoverflow.com/questions/8063228/how-do-i-check-if-a-variable-exists-in-a-list-in-bash
contains() {
  local item="$1"
  shift  # Shift all arguments to the left (original $1 gets lost)
  local list=("$@")
  # echo ${list[@]}
  # echo $item
  local result=0
  for i in ${list[@]}; do
    # if [[ $i == $item ]]; then
    if [[ "$i" == "$item" ]]; then
      result=1
    fi
  done
  echo $result
}



# SCRIPT STARTS HERE
# ==============================================================================

# Go to output anat folder, where most of the outputs will be located
cd ${ofolder_reg}

# TO_EXCLUDE=(
#   "sub-amu01_acq-MTon_MTS"
#   "sub-amu03_acq-MTon_MTS"
# )

echo -e ${TO_EXCLUDE[@]}

for i in ${!FILES_SRC[@]}; do
  exclude_file=`contains ${FILES_DEST[$i]} "${TO_EXCLUDE[@]}"`
  # echo "i: $i, ${FILES_DEST[$i]}, $exclude_file"
  if [[ $exclude_file -eq 1 ]]; then
    echo -e "\nWARNING: File excluded: ${FILES_DEST[$i]}.nii.gz"
  else
    # Copy and rename file
    cp tmp/${FILES_SRC[$i]}.nii.gz ${FILES_DEST[$i]}.nii.gz
    # Duplicate segmentation to be used by other contrasts
    cp tmp/${file_t1w_mts}_crop_r_seg-manual.nii.gz ${ofolder_seg}/${FILES_DEST[$i]}_seg-manual.nii.gz
    # Remove empty slices at the edge
    prepdata -i ${FILES_DEST[$i]}.nii.gz -s ${ofolder_seg}/${FILES_DEST[$i]}_seg-manual.nii.gz remove-slice
    # Generate final QC
    sct_qc -i ${FILES_DEST[$i]}.nii.gz -s ${ofolder_seg}/${FILES_DEST[$i]}_seg-manual.nii.gz -p sct_deepseg_sc -qc ${PATH_QC}2 -qc-subject ${SUBJECT}
    # Copy json file and rename them
    cp ${PATH_IN}/${FILES_DEST[$i]}.json ${FILES_DEST[$i]}.json
  fi
done

# TODO: Copy the following json files manually:
# cp ${PATH_IN}/../../dataset_description.json ../../
# cp ${PATH_IN}/../../participants.json ../../
# cp ${PATH_IN}/../../participants.tsv ../../
