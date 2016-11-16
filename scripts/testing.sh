#!/bin/bash

CORES=2

for i in {0..0} ; do
    ./compileGraph.py -c $CORES -s $i
done
