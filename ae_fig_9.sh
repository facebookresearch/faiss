#!/bin/bash
cd build
make Fig_9

export OMP_NUM_THREADS=16
ae_asplos25/Fig_9

