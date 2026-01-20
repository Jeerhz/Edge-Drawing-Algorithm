#!/bin/bash
# Arguments: scales (>=0)
# -gradient: min gradient 
# -gap: anchor gap
# -length: min length
# -sigma: Gaussian blur

grad=$1
gap=$2
length=$3
sigma=$4

$bin/build/edgeDrawing -g $grad -a $gap -l $length -s $sigma $input_0 edges.png
