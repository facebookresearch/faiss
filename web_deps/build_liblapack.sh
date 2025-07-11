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

if [ -d "./lapack-3.12.0" ]; then
    rm -rf lapack-3.12.0
fi
if [ ! -f "./v3.12.0.tar.gz" ]; then
    curl -LO https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.12.0.tar.gz
fi
tar zxf v3.12.0.tar.gz
rm -f v3.12.0.tar.gz

cd lapack-3.12.0
cp make.inc.example make.inc
sed -i \
    -e "s/^FC[[:space:]]*=.*/FC = $EMFC/" \
    -e "s/^FFLAGS[[:space:]]*=.*/FFLAGS = -O2/" \
    -e "s/^FFLAGS_DRV[[:space:]]*=.*/FFLAGS_DRV = \$(FFLAGS)/" \
    -e "s/^FFLAGS_NOOPT[[:space:]]*=.*/FFLAGS_NOOPT = -O0/" \
    -e "s/^AR[[:space:]]*=.*/AR = $EMAR/" \
    -e "s/^RANLIB[[:space:]]*=.*/RANLIB = $EMRANLIB/" \
    -e "s/^TIMER[[:space:]]*=.*/TIMER = INT_CPU_TIME/" \
    "make.inc"
make lib
if [ -e "./liblapack.a" ]; then
    cp ./liblapack.a ../liblapack.a
    cd -
    ls liblapack.a
else
    echo "Build Failed. Cannot find built static library."
    cd -
    exit 1
fi
rm -rf lapack-3.12.0