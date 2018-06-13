AC_DEFUN([FA_PYTHON], [

AC_ARG_WITH(python,
  [AS_HELP_STRING([--with-python=<bin>], [use Python binary <bin>])])
case $with_python in
  "") PYTHON_BIN=python ;;
  *) PYTHON_BIN="$with_python"
esac

AC_CHECK_PROG(PYTHON, $PYTHON_BIN, $PYTHON_BIN)
fa_python_bin=$PYTHON

AC_ARG_WITH(python-config,
  [AS_HELP_STRING([--with-python-config=<bin>], [use Python config binary <bin>])])
case $with_python_config in
  "") PYTHON_CFG_BIN="${PYTHON_BIN}-config" ;;
  *) PYTHON_CFG_BIN="$with_python_config"
esac

AC_CHECK_PROG(PYTHON_CFG, $PYTHON_CFG_BIN, $PYTHON_CFG_BIN)
fa_python_cfg_bin=$PYTHON_CFG

if test x$fa_python_cfg_bin != x; then
  AC_MSG_CHECKING([for Python C flags])
  fa_python_cflags=`${PYTHON_CFG} --includes`
  if test x"$fa_python_cflags" == x; then
    AC_MSG_RESULT([not found])
    AC_MSG_WARN([You won't be able to build the python interface.])
  else
    AC_MSG_RESULT($fa_python_cflags)
    AC_SUBST(PYTHON_CFLAGS, $fa_python_cflags)
  fi

  AC_MSG_CHECKING([for Python ld flags])
  fa_python_ldflags=`${PYTHON_CFG} --ldflags`
  if test x"$fa_python_ldflags" == x; then
    AC_MSG_RESULT([not found])
  else
    AC_MSG_RESULT($fa_python_ldflags)
    AC_SUBST(PYTHON_LDFLAGS, $fa_python_ldflags)
  fi
else
  AC_MSG_WARN([You won't be able to build the python interface.])
fi
])
