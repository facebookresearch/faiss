#include <cstdio>

template <int>
void func();

template <>
void func<1>() {
    printf("func 1\n");
}

template <>
void func<2>() {
    printf("func 2\n");
}
