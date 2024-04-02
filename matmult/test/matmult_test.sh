#!/bin/bash

# Check if at least one argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: source matmult_test.sh <matmult version>"
    return 1
fi

matmult_version="$1"

# Define a list of input arguments
inputArguments=(128 256 512 1024 2048 4096)

# Loop through all input arguments in the list
for i in "${inputArguments[@]}"
do
   ../bin/matmult_"$matmult_version" "$i" >> results/matmult_"$matmult_version"_results.txt
done
