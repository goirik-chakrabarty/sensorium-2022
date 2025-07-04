#!/bin/bash

# Define variables
PROJECT_ROOT="$PWD"
SIF_IMAGE="$PWD/sensorium.sif"
ENV_FILE=".env"

# # Export environment variables from .env file
# set -o allexport
# source "$ENV_FILE"
# set +o allexport

# Run the container
apptainer exec \
 --nv \
 --bind "$PROJECT_ROOT:/project" \
 --bind "$PROJECT_ROOT/notebooks:/notebooks" \
 "$SIF_IMAGE" \
 bash run_train.sh