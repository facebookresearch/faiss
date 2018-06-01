AC_DEFUN([FA_PROG_SWIG], [

AC_ARG_WITH(swig,
[AS_HELP_STRING([--with-swig=<bin>], [use SWIG binary <bin>])])
case $with_swig in
  "") SWIG_BIN=swig ;;
  *) SWIG_BIN="$with_swig"
esac

AC_CHECK_PROG(SWIG, $SWIG_BIN, $SWIG_BIN)
AC_SUBST(SWIG)
])
