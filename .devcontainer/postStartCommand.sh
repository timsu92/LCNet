#!/bin/bash
# exits if sub-processes fail, or if an unset variable is attempted to be used, or if there's a pipe failure
set -euo pipefail
script_dir="$(dirname "${BASH_SOURCE[0]:-$0}")"

if [ ! -e "$script_dir/../data/cifar-10-batches-py" ] && [ ! -e "$script_dir/../data/cifar-10-python.tar.gz" ]; then
    echo "Downloading CIFAR-10 dataset..."
    mkdir -p "$script_dir/../data"
    wget -O "$script_dir/../data/cifar-10-python.tar.gz" https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    echo "CIFAR-10 dataset downloaded."
fi
