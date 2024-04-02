#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: ./conv_test.sh <conv version>"
    return
fi

conv_version="$1"

# input vector sizes
input_vector_sizes=(512 1024 4096 16384)

# Loop through all input vector sizes
for input_size in "${input_vector_sizes[@]}"
do
    echo "running input size: $input_size for kernel number: $conv_version"
    # formatting for parsing
    echo "INPUT: $input_size, FILTER: 512 ######################################################################################################" >> results/conv_"$conv_version"_results.txt
    # call binary with input size -> appending results to a file
    ncu ./bin/conv_"$conv_version" "$input_size" >> results/conv_"$conv_version"_results.txt
    echo "TIMING ###############################################################################################################################" >> results/conv_"$conv_version"_results.txt
    ./bin/conv_"$conv_version" "$input_size" >> results/conv_"$conv_version"_results.txt
done
