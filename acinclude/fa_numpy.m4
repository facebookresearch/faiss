AC_DEFUN([FA_NUMPY], [
AC_REQUIRE([FA_PYTHON])

AC_MSG_CHECKING([for numpy headers path])

fa_numpy_headers=`$PYTHON_BIN -c "import numpy; print(numpy.get_include())"`

if test x$fa_numpy_headers == x; then
   AC_MSG_RESULT([not found])
   AC_MSG_WARN([You won't be able to build the python interface.])
else
  AC_MSG_RESULT($fa_numpy_headers)
  AC_SUBST(NUMPY_INCLUDE, $fa_numpy_headers)
fi
])dnl
