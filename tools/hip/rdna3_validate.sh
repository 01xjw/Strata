#!/usr/bin/env bash
# Build and run the model-free RDNA3 validation set.  Model/quality A/Bs are documented in docs/RDNA3_VALIDATION.md.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ACTION="${1:-all}"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build-rdna3}"
STRATA_HIP_ARCH="${STRATA_HIP_ARCH:-gfx1100}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
JOBS="${JOBS:-$(nproc)}"

usage() {
    cat <<'EOF'
usage: tools/hip/rdna3_validate.sh [configure|build|host-tests|hip-tests|all]

Environment:
  BUILD_DIR        CMake build directory (default: build-rdna3)
  STRATA_HIP_ARCH  HIP architecture or semicolon list (default: gfx1100)
  ROCM_PATH        ROCm installation (default: /opt/rocm)
  STRATA_GGML_DIR  Optional offline checkout of the repository-pinned llama.cpp
  JOBS             Parallel build jobs
EOF
}

case "${ACTION}" in
    configure|build|host-tests|hip-tests|all) ;;
    -h|--help|help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

configure() {
    local args=(
        -S "${ROOT}" -B "${BUILD_DIR}"
        -DCMAKE_BUILD_TYPE=Release
        -DSTRATA_ENABLE_HIP=ON
        -DSTRATA_ENABLE_CUDA=OFF
        -DSTRATA_BUILD_TESTS=ON
        -DSTRATA_PREFILL_MMQ=ON
        "-DCMAKE_HIP_ARCHITECTURES=${STRATA_HIP_ARCH}"
    )
    if [[ -x "${ROCM_PATH}/llvm/bin/clang++" ]]; then
        args+=("-DCMAKE_HIP_COMPILER=${ROCM_PATH}/llvm/bin/clang++")
    fi
    if [[ -n "${STRATA_GGML_DIR:-}" ]]; then
        args+=("-DSTRATA_GGML_DIR=${STRATA_GGML_DIR}")
    fi
    cmake "${args[@]}"
}

build_all() {
    cmake --build "${BUILD_DIR}" -j"${JOBS}"
}

host_tests() {
    cmake --build "${BUILD_DIR}" --target gfx_arch_test arch_defaults_test -j"${JOBS}"
    ctest --test-dir "${BUILD_DIR}" --output-on-failure -R '^(gfx_arch_test|arch_defaults_test)$'
}

hip_tests() {
    "${BUILD_DIR}/strata-device" --list-devices
    "${BUILD_DIR}/strata-device" --selftest
    ctest --test-dir "${BUILD_DIR}" -N
    ctest --test-dir "${BUILD_DIR}" --output-on-failure --timeout 600 \
        -R '^(hip_.*|qsa_select_bench|qsa_select_bench_fp32_tiled)$'
}

case "${ACTION}" in
    configure) configure ;;
    build) build_all ;;
    host-tests) host_tests ;;
    hip-tests) hip_tests ;;
    all)
        configure
        build_all
        host_tests
        hip_tests
        ;;
esac
