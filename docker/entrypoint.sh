#!/bin/bash
set -e

echo "Initializing Conda..."
source /opt/conda/etc/profile.d/conda.sh

echo "Activating Conda environment 'gaussianpro'..."
conda activate gaussianpro

echo "Conda environment activated: $(conda info --envs | grep '*' )"

if [ "$#" -gt 0 ]; then
    echo "Executing command: $@"
    exec "$@"
else
    echo "Starting interactive bash shell..."
    exec bash
fi
