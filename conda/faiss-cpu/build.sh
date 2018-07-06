./configure
make
cd python
make
$PYTHON setup.py install --single-version-externally-managed --record=record.txt
