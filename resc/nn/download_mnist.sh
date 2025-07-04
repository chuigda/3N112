#!/usr/bin/env bash

# As is observed by Chuigda, Yann Lecun's website has exploded recently. We have to use other
# channels for accessing these datasets.

wget https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
wget https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
wget https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
wget https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz

# Decompress the downloaded files, output file with .bin extension
gunzip -c train-images-idx3-ubyte.gz > train-images-idx3-ubyte.bin
gunzip -c train-labels-idx1-ubyte.gz > train-labels-idx1-ubyte.bin
gunzip -c t10k-images-idx3-ubyte.gz > t10k-images-idx3-ubyte.bin
gunzip -c t10k-labels-idx1-ubyte.gz > t10k-labels-idx1-ubyte.bin
