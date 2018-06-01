AC_DEFUN([FA_CHECK_CUDA], [

AC_ARG_WITH(cuda,
[AS_HELP_STRING([--with-cuda=<prefix>], [prefix of the CUDA installation])])
case $with_cuda in
"") cuda_prefix=/usr/local/cuda ;;
*) cuda_prefix="$with_cuda"
esac

AC_CHECK_PROG(NVCC, "nvcc", "$cuda_prefix/bin/nvcc", "", "$cuda_prefix/bin")
fa_nvcc_bin=$NVCC

if test x$fa_nvcc_bin != x; then
  fa_save_CPPFLAGS="$CPPFLAGS"
  fa_save_LDFLAGS="$LDFLAGS"
  fa_save_LIBS="$LIBS"

  NVCC_CPPFLAGS="-I$cuda_prefix/include"
  NVCC_LDFLAGS="-L$cuda_prefix/lib"

  CPPFLAGS="$NVCC_CPPFLAGS $CPPFLAGS"
  LDFLAGS="$NVCC_LDFLAGS $LDFLAGS"

  AC_CHECK_HEADER([cuda.h], [], AC_MSG_FAILURE([Couldn't find cuda.h]))
  AC_CHECK_LIB([cuda], [cuInit], [], AC_MSG_FAILURE([Couldn't find libcuda]))

  CPPFLAGS="$fa_save_CPPFLAGS"
  LDFLAGS="$fa_save_LDFLAGS"
  LIBS="$fa_save_LIBS"
  NVCC_LIBS="-Xlinker \"${LIBS}\" -lcuda"
  NVCC_LDFLAGS="-Xlinker \"${LDFLAGS}\""
else
  cuda_prefix=""
fi

AC_SUBST(NVCC)
AC_SUBST(NVCC_CPPFLAGS)
AC_SUBST(NVCC_LDFLAGS)
AC_SUBST(NVCC_LIBS)
AC_SUBST(CUDA_PREFIX, $cuda_prefix)
])
