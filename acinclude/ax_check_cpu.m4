# serial 1

AC_DEFUN([AX_CPU_ARCH], [

AC_MSG_CHECKING([for cpu arch])
AC_CANONICAL_TARGET

case $target in
  amd64-* | x86_64-*)
    ARCH_CPUFLAGS="-mpopcnt"
    ARCH_CXXFLAGS="-m64"

    AX_GCC_X86_CPU_SUPPORTS(avx2,
        [ARCH_CPUFLAGS+=" -mavx2 -mf16c"],
        [ARCH_CPUFLAGS+=" -msse4"])
    ;;
  aarch64*-*)
dnl This is an arch for Nvidia Xavier a proper detection would be nice.
    ARCH_CPUFLAGS="-march=armv8.2-a"
    ;;
  *) ;;
esac
AC_MSG_RESULT([$target CPUFLAGS+=$ARCH_CPUFLAGS CXXFLAGS+=$ARCH_CXXFLAGS])

AC_SUBST(ARCH_CPUFLAGS)
AC_SUBST(ARCH_CXXFLAGS)

])dnl
