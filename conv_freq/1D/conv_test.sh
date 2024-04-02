#!/bin/bash

# Check if at least one argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: ./conv_test.sh <conv version>"
    return
fi

conv_version="$1"

make conv_$conv_version

# Define a list of input vector sizes
input_vector_sizes=(128 1024 4096 16384)
# Define a list of filter vector sizes
filter_vector_sizes=(8 16 32 64 128  256)

# Loop through all input vector sizes
for input_size in "${input_vector_sizes[@]}"
do
    echo "Running Kernel Input of $input_size"

    # Nested loop through all filter vector sizes
    for filter_size in "${filter_vector_sizes[@]}"
    do
        # Call your binary with both input and filter sizes, appending results to a file
        ./bin/conv_"$conv_version" "$input_size" "$filter_size" >> results/conv_"$conv_version"_results.txt
        # Breaking up results for easier parsing
        echo "________________________________________________________________________________________________" >> results/conv_"$conv_version"_results.txt
    done
done
