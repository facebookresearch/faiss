#!/bin/sh

PYTHON=${PYTHON:-"python"}

make

for t in test_*.py; do
    PYTHONPATH=.. $PYTHON $t || exit 1
done

for t in test_*.cpp; do
    ./${t%.*} || exit 1
done
