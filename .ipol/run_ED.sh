#!/bin/bash
# Arguments: scales (>=0)
# -gradient: min gradient 
# -gap: anchor gap
# -length: min length
# -sigma: Gaussian blur
# -valid: a conrario validation

grad=$1
gap=$2
length=$3
sigma=$4
valid=$5
valid=${valid/true/-e 0}
valid=${valid/false/}

$bin/build/edgeDrawing $valid -g $grad -a $gap -l $length -s $sigma $input_0 edges.png
