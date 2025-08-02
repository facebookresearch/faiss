EMFC="${FC:-flang}"
EMAR="${AR:-emar}"
EMRANLIB="${RANLIB:-emranlib}"

if ! command -v ${EMFC} &> /dev/null; then
  echo "Build Failed. Cannot execute Fortran compiler: ${EMFC}"
  exit 1
fi

if ! command -v ${EMAR} &> /dev/null; then
  echo "Build Failed. Cannot execute Archiver: ${EMAR}"
  exit 1
fi

if ! command -v ${EMRANLIB} &> /dev/null; then
  echo "Build Failed. Cannot execute ranlib: ${EMRANLIB}"
  exit 1
fi

if [ -d "./BLAS-3.12.0" ]; then
    rm -rf BLAS-3.12.0
fi
if [ ! -f "./blas-3.12.0.tgz" ]; then
    curl -LO https://www.netlib.org/blas/blas-3.12.0.tgz
fi
tar zxf blas-3.12.0.tgz
rm -f blas-3.12.0.tgz

cd BLAS-3.12.0
sed -i \
    -e "s/^FC[[:space:]]*=.*/FC = ${EMFC}/" \
    -e "s/^FFLAGS[[:space:]]*=.*/FFLAGS = -O2/" \
    -e "s/^FFLAGS_DRV[[:space:]]*=.*/FFLAGS_DRV = \$(FFLAGS)/" \
    -e "s/^FFLAGS_NOOPT[[:space:]]*=.*/FFLAGS_NOOPT = -O0/" \
    -e "s/^AR[[:space:]]*=.*/AR = $EMAR/" \
    -e "s/^RANLIB[[:space:]]*=.*/RANLIB = $EMRANLIB/" \
    "make.inc"
make

if [ -e "./blas_LINUX.a" ]; then
    cp ./blas_LINUX.a ../libblas.a
    cd -
    ls libblas.a
else
    echo "Build Failed. Cannot find built static library."
    cd -
    exit 1
fi
rm -rf BLAS-3.12.0