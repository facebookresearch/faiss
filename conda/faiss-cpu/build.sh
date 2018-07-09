# NOTE(beauby): This is needed as conda does not symlink `python-config`.
./configure --with-python-config=python${PY_VER}-config
make -j $CPU_COUNT
cd python
make
$PYTHON setup.py install --single-version-externally-managed --record=record.txt
