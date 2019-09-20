# serial 1

AC_DEFUN([AX_CPU_ARCH], [

AC_MSG_CHECKING([for cpu arch])

  AC_CANONICAL_TARGET

  case $target in
    amd64-* | x86_64-*)
      ARCH_CPUFLAGS="-mpopcnt -msse4"
      ARCH_CXXFLAGS="-m64";;
    *) ;;
  esac

AC_MSG_RESULT([$target CPUFLAGS+="$ARCH_CPUFLAGS" CXXFLAGS+="$ARCH_CXXFLAGS"])

AC_SUBST(ARCH_CPUFLAGS)
AC_SUBST(ARCH_CXXFLAGS)

])dnl
