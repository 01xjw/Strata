// src/kernels/gfx_arch_test.cpp - host-only checks for exact AMD architecture-family gates.
#include "strata/kernels/gfx_arch.hpp"

#include <cstdio>
#include <initializer_list>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main() {
    using namespace strata::kernels;

    CHECK(gfx_arch_is("gfx1100", "gfx1100"));
    CHECK(gfx_arch_is("gfx1100:sramecc-:xnack-", "gfx1100"));
    CHECK(!gfx_arch_is("gfx11000", "gfx1100"));
    CHECK(!gfx_arch_is("gfx1100x", "gfx1100"));
    CHECK(!gfx_arch_is(nullptr, "gfx1100"));

    for (const char* a : {"gfx1100", "gfx1101", "gfx1102", "gfx1102:sramecc-:xnack-"})
        CHECK(gfx_arch_is_rdna3_dgpu(a));
    for (const char* a : {"gfx1103", "gfx1150", "gfx1151", "gfx1152", "gfx1200", "gfx11020", ""})
        CHECK(!gfx_arch_is_rdna3_dgpu(a));

    for (const char* a : {"gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151:sramecc-:xnack-"})
        CHECK(gfx_arch_is_gfx11_wmma(a));
    for (const char* a : {"gfx1152", "gfx1200", "gfx11510", ""})
        CHECK(!gfx_arch_is_gfx11_wmma(a));

    for (const char* a : {"gfx1200", "gfx1201", "gfx1201:sramecc-:xnack-"}) {
        CHECK(gfx_arch_is_gfx12_wmma(a));
        CHECK(gfx_arch_is_gfx12(a));
    }
    for (const char* a : {"gfx1100", "gfx1202", "gfx12010", ""}) {
        CHECK(!gfx_arch_is_gfx12_wmma(a));
        CHECK(!gfx_arch_is_gfx12(a));
    }

    CHECK(gfx_arch_is_gfx1151("gfx1151"));
    CHECK(gfx_arch_is_gfx1151("gfx1151:sramecc-:xnack-"));
    CHECK(!gfx_arch_is_gfx1151("gfx11510"));

    std::printf(fails ? "gfx_arch_test: %d FAILED\n" : "gfx_arch_test: OK\n", fails);
    return fails ? 1 : 0;
}
