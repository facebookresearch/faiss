#!/bin/sh

PYTHON=${PYTHON:-"python"}

make

for t in test_*.py; do
    PYTHONPATH=.. $PYTHON $t || exit 1
done

for t in test_*.cpp; do
    if [ "$t" != "test_blas.cpp" ]; then
        ./${t%.*} || exit 1
    fi
done
