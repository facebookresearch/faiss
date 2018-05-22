AC_DEFUN([FA_PYTHON], [
AC_MSG_CHECKING(for python build information)

AC_ARG_WITH(python,
  [AS_HELP_STRING([--with-python=<bin>], [use Python library <bin>])])
case $with_python in
  "") PYTHON_BIN=python ;;
  *) PYTHON_BIN="$with_python"
esac

AC_CHECK_PROG(PYTHON, $PYTHON_BIN, $PYTHON_BIN)

AC_MSG_CHECKING([for Python headers path])

fa_python_headers=`$PYTHON_BIN -c "from distutils.sysconfig import *; print(get_python_inc())"`

if test x$fa_python_headers == x; then
  AC_MSG_RESULT([not found])
  AC_MSG_WARN([You won't be able to build the python interface.])
else
  AC_MSG_RESULT($fa_python_headers)
  AC_SUBST(PYTHON_INCLUDE, $fa_python_headers)
fi
])dnl
