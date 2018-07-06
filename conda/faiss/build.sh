./configure
make
make -C gpu
cd python
make gpu
make
$PYTHON setup.py install --single-version-externally-managed --record=record.txt
