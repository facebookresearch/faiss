./configure
make -j $CPU_COUNT
make -C gpu -j $CPU_COUNT
cd python
make gpu
make
$PYTHON setup.py install --single-version-externally-managed --record=record.txt
