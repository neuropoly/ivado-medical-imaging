#!/bin/bash
#
# Clean if files were mistakenely generated in root folder
#
# Usage:
#   ./clean.sh

# Immediately exit if error
set -v

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# Retrieve input params
SUBJECT=$1

cd ${SUBJECT}/anat || exit
rm straightening.cache
rm ${SUBJECT}_acq-T1w_MTS_seg.nii.gz
rm ${SUBJECT}_T1w_seg.nii.gz
rm ${SUBJECT}_T2star_seg.nii.gz
rm ${SUBJECT}_T2w_seg.nii.gz
