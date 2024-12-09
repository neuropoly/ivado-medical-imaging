#!/bin/bash
#
# Delete temporary files. Run this script once all files have been verified.
# Usage:
#   ./delete_temp_files.sh <FILE_PARAM>

# Uncomment for full verbose
#set -v

# Immediately exit if error
set -e

# Retrieve input params
SUBJECT=$1

# Create BIDS architecture
ofolder_reg="${PATH_DATA_PROCESSED}/${SUBJECT}/anat"

# Go to output anat folder, where most of the outputs will be located
cd ${ofolder_reg}

# Remove tmp folder
rm -rf tmp
