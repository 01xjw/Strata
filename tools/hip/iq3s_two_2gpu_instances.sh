#!/usr/bin/env bash
# Bare-metal helper for two production-style IQ3_S Strata services on gfx1100.
#
# It prepares the offline pack from already-downloaded GGUF shards, writes two
# two-GPU configs, starts one server per GPU pair, and can run a bounded
# concurrency sweep against both ports.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ACTION="${1:-help}"

DATA_ROOT="${DATA_ROOT:-/root/strata-data}"
MODEL_DIR="${MODEL_DIR:-${DATA_ROOT}/models/Qwen3.8-flash-Next/IQ3_S}"
PACK_DIR="${PACK_DIR:-${DATA_ROOT}/packs/qwen3.8-flash-next-iq3_s}"
RUN_DIR="${RUN_DIR:-${DATA_ROOT}/runs/iq3s-two-2gpu}"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build-rdna3}"
STRATA_HIP_ARCH="${STRATA_HIP_ARCH:-gfx1100}"
ROCM_PATH="${ROCM_PATH:-}"
JOBS="${JOBS:-$(nproc)}"

HOST="${HOST:-127.0.0.1}"
PORT_A="${PORT_A:-8090}"
PORT_B="${PORT_B:-8091}"
GPU_A="${GPU_A:-0,1}"
GPU_B="${GPU_B:-2,3}"
CONTEXT="${CONTEXT:-65536}"
PARALLEL="${PARALLEL:-8}"
POOL_WORKERS="${POOL_WORKERS:-24}"
VRAM_RESERVE_MIB="${VRAM_RESERVE_MIB:-2048}"
SPEC="${SPEC:-4}"
SPEC_MIN_P="${SPEC_MIN_P:-0.5}"
MTP="${MTP:-1}"
MTP_DIR="${MTP_DIR:-${DATA_ROOT}/mtp/rt}"
PIPELINE_WINDOWS="${PIPELINE_WINDOWS:-}"
HUGEPAGE_MALLOC="${HUGEPAGE_MALLOC:-1}"
SLOT_COPY_DIRECT="${SLOT_COPY_DIRECT:-1}"
SERVER_ENV="${SERVER_ENV:-}"
API_KEY="${STRATA_API_KEY:-${API_KEY:-}}"

SHARD1="${MODEL_DIR}/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
SHARD2="${MODEL_DIR}/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf"
ENGINE="${ENGINE:-${BUILD_DIR}/strata}"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=python3
fi

resolve_rocm_path() {
  if [[ -n "${ROCM_PATH}" && -d "${ROCM_PATH}" ]]; then
    return
  fi
  if [[ -f /opt/rocm/lib/cmake/hip/hip-config.cmake ]]; then
    ROCM_PATH=/opt/rocm
    return
  fi
  local candidate
  candidate="$(ls -d /opt/rocm-* 2>/dev/null | sort -V | tail -1 || true)"
  if [[ -n "${candidate}" && -f "${candidate}/lib/cmake/hip/hip-config.cmake" ]]; then
    ROCM_PATH="${candidate}"
    return
  fi
  ROCM_PATH=/opt/rocm
}

resolve_ggml_dir() {
  if [[ -n "${STRATA_GGML_DIR:-}" ]]; then
    return
  fi
  if [[ -f "${ROOT}/third_party/llama.cpp/CMakeLists.txt" ]]; then
    export STRATA_GGML_DIR="${ROOT}/third_party/llama.cpp"
  elif [[ -f "${DATA_ROOT}/src/llama.cpp/CMakeLists.txt" ]]; then
    export STRATA_GGML_DIR="${DATA_ROOT}/src/llama.cpp"
  fi
}

usage() {
  cat <<'EOF'
usage: tools/hip/iq3s_two_2gpu_instances.sh ACTION

Actions:
  prepare       Build the HIP engine if needed, create pack/tokenizer, write configs
  start         Start two services in the background
  wait          Wait until both services answer /v1/models
  smoke         Send one small request to each service
  load          Run short concurrency sweeps against both services
  long-load     Run long-context concurrency sweeps against both services
  stop          Stop services started by this script
  status        Show PIDs, ports and recent logs
  all           prepare, start, wait, smoke, load

Useful environment:
  DATA_ROOT=/root/strata-data
  MODEL_DIR=<dir holding the two IQ3_S GGUF shards>
  GPU_A=0,1 GPU_B=2,3
  PORT_A=8090 PORT_B=8091
  CONTEXT=65536 PARALLEL=8
  SPEC=4 SPEC_MIN_P=0.5
  MTP_DIR=<draft head runtime dir, default $DATA_ROOT/mtp/rt; used when it holds experts.bin; MTP=0 skips it>
  PIPELINE_WINDOWS=1            overlap the two GPUs on short prompt reads (empty: the engine default, off)
  HUGEPAGE_MALLOC=1             GLIBC_TUNABLES=glibc.malloc.hugetlb=1 for the servers (0: glibc default)
  SLOT_COPY_DIRECT=1            STRATA_SLOT_COPY_DIRECT=1 STRATA_SLOT_COPY_BATCHSYNC=1: admissions copy slots on the GPU
                                instead of through host memory (0: the engine default, the host path)
  SERVER_ENV="STRATA_X=1 ..."   extra environment for both servers
  ENGINE=<strata binary, default $BUILD_DIR/strata>
  STRATA_API_KEY=<optional bearer key>
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

require_files() {
  [[ -s "${SHARD1}" ]] || die "missing ${SHARD1}"
  [[ -s "${SHARD2}" ]] || die "missing ${SHARD2}"
  [[ -s "${ROOT}/data/expert-profile.bin" ]] || die "missing ${ROOT}/data/expert-profile.bin"
}

configure_engine() {
  if [[ -x "${ENGINE}" ]]; then
    return
  fi
  resolve_rocm_path
  resolve_ggml_dir
  local args=(
    -S "${ROOT}" -B "${BUILD_DIR}"
    -DCMAKE_BUILD_TYPE=Release
    -DSTRATA_ENABLE_HIP=ON
    -DSTRATA_ENABLE_CUDA=OFF
    -DSTRATA_PREFILL_MMQ=ON
    "-DCMAKE_HIP_ARCHITECTURES=${STRATA_HIP_ARCH}"
    "-DCMAKE_PREFIX_PATH=${ROCM_PATH}${CMAKE_PREFIX_PATH:+;${CMAKE_PREFIX_PATH}}"
    "-Dhip_DIR=${ROCM_PATH}/lib/cmake/hip"
  )
  if [[ -x "${ROCM_PATH}/llvm/bin/clang++" ]]; then
    args+=("-DCMAKE_HIP_COMPILER=${ROCM_PATH}/llvm/bin/clang++")
  fi
  if [[ -n "${STRATA_GGML_DIR:-}" ]]; then
    args+=("-DSTRATA_GGML_DIR=${STRATA_GGML_DIR}")
  fi
  cmake "${args[@]}"
  cmake --build "${BUILD_DIR}" --target strata -j"${JOBS}"
}

prepare_pack() {
  if [[ -s "${PACK_DIR}/native_experts.txt" && -s "${PACK_DIR}/index.txt" && -s "${PACK_DIR}/tokenizer/vocab.json" ]]; then
    if [[ -s "${PACK_DIR}/experts.bin" ]]; then
      echo "pack exists: ${PACK_DIR}"
      return
    fi
    echo "pack exists but experts.bin is missing; rebuilding experts.bin"
  fi
  mkdir -p "${PACK_DIR}"
  "${PYTHON}" "${ROOT}/tools/iq_pack.py" --gguf "${SHARD1}" --out "${PACK_DIR}" --experts-bin
}

json_array() {
  local csv="$1"
  local out="["
  local first=1
  local x
  IFS=',' read -r -a parts <<<"${csv}"
  for x in "${parts[@]}"; do
    x="${x//[[:space:]]/}"
    [[ -n "${x}" ]] || continue
    if [[ "${first}" == 0 ]]; then out+=", "; fi
    out+="${x}"
    first=0
  done
  out+="]"
  printf '%s' "${out}"
}

write_config() {
  local name="$1" gpus="$2" port="$3"
  local cfg="${RUN_DIR}/${name}.json"
  local extra_args=""
  if [[ "${MTP}" != 0 && -s "${MTP_DIR}/experts.bin" ]]; then
    extra_args=",
    \"--mtp\", \"${MTP_DIR}\""
  elif [[ "${MTP}" != 0 ]]; then
    echo "warning: no MTP draft head at ${MTP_DIR}; --spec runs without drafts" >&2
  fi
  if [[ -n "${PIPELINE_WINDOWS}" ]]; then
    extra_args+=",
    \"--pipeline-windows\", \"${PIPELINE_WINDOWS}\""
  fi
  mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/profiles" "${RUN_DIR}/results"
  cat >"${cfg}" <<EOF
{
  "exe": "${ENGINE}",
  "cwd": "${ROOT}",
  "backend": "hip",
  "gpu": $(json_array "${gpus}"),
  "gpu_order": "as_given",
  "host": "${HOST}",
  "port": ${port},
  "model_name": "qwen3.8-flash-next-iq3_s-${name}",
  "tokenizer": "${PACK_DIR}/tokenizer",
  "parallel": ${PARALLEL},
  "api_key": "${API_KEY}",
  "args": [
    "--pack", "${PACK_DIR}",
    "--native", "${SHARD1}",
    "--ple-gguf", "${SHARD2}",
    "--mmap-experts",
    "--expert-profile", "${ROOT}/data/expert-profile.bin",
    "--expert-profile-save", "${RUN_DIR}/profiles/${name}-expert-profile.bin",
    "--expert-cache", "auto",
    "--prefill", "auto",
    "--spec", "${SPEC}",
    "--spec-min-p", "${SPEC_MIN_P}",
    "--max-context", "${CONTEXT}",
    "--kv", "int8",
    "--pool-workers", "${POOL_WORKERS}",
    "--vram-reserve-mib", "${VRAM_RESERVE_MIB}"${extra_args}
  ],
  "log": "${RUN_DIR}/logs/${name}-engine.log"
}
EOF
  echo "${cfg}"
}

prepare() {
  require_files
  configure_engine
  prepare_pack
  write_config "a" "${GPU_A}" "${PORT_A}" >/dev/null
  write_config "b" "${GPU_B}" "${PORT_B}" >/dev/null
  echo "configs: ${RUN_DIR}/a.json ${RUN_DIR}/b.json"
}

auth_args=()
auth_header=()
if [[ -n "${API_KEY}" ]]; then
  auth_args=(--key "${API_KEY}")
  auth_header=(-H "Authorization: Bearer ${API_KEY}")
fi

# A container whose PID 1 does not reap leaves exited servers as zombies: kill -0 still succeeds on those.
alive() {
  local stat
  stat="$(ps -o stat= -p "$1" 2>/dev/null)" || return 1
  [[ -n "${stat}" && "${stat}" != Z* ]]
}

server_pid() {
  local name="$1"
  local pidfile="${RUN_DIR}/${name}.pid"
  [[ -s "${pidfile}" ]] || return 1
  local pid
  pid="$(cat "${pidfile}")"
  alive "${pid}" || return 1
  printf '%s' "${pid}"
}

start_one() {
  local name="$1" port="$2"
  local cfg="${RUN_DIR}/${name}.json"
  [[ -s "${cfg}" ]] || die "missing ${cfg}; run prepare first"
  if server_pid "${name}" >/dev/null; then
    echo "${name}: already running pid $(server_pid "${name}")"
    return
  fi
  # Each request's conversation checkpoint is ~58 MB of fresh host memory per GPU stage: on 4 KiB pages its faults
  # cost ~25 ms and stalled prompt reads behind them; glibc's hugetlb tunable backs those blocks with THP (madvise).
  local run_env=()
  if [[ "${HUGEPAGE_MALLOC}" != 0 && -z "${GLIBC_TUNABLES:-}" && "${SERVER_ENV}" != *GLIBC_TUNABLES=* ]]; then
    run_env=(GLIBC_TUNABLES=glibc.malloc.hugetlb=1)
  fi
  if [[ "${SLOT_COPY_DIRECT}" != 0 && "${SERVER_ENV}" != *STRATA_SLOT_COPY_DIRECT=* ]]; then
    run_env+=(STRATA_SLOT_COPY_DIRECT=1 STRATA_SLOT_COPY_BATCHSYNC=1)
  fi
  # shellcheck disable=SC2086
  env "${run_env[@]}" ${SERVER_ENV} nohup "${PYTHON}" -m serve.server --engine strata --config "${cfg}" --host "${HOST}" \
    --port "${port}" </dev/null >"${RUN_DIR}/logs/${name}-server.log" 2>&1 &
  echo "$!" >"${RUN_DIR}/${name}.pid"
  echo "${name}: started pid $(cat "${RUN_DIR}/${name}.pid") on ${HOST}:${port}"
}

start() {
  start_one a "${PORT_A}"
  start_one b "${PORT_B}"
}

stop_one() {
  local name="$1"
  local pidfile="${RUN_DIR}/${name}.pid"
  if server_pid "${name}" >/dev/null; then
    local pid
    pid="$(cat "${pidfile}")"
    kill "${pid}" || true
    for _ in {1..60}; do
      alive "${pid}" || break
      sleep 1
    done
    alive "${pid}" && kill -9 "${pid}" 2>/dev/null || true
    echo "${name}: stopped ${pid}"
  else
    echo "${name}: not running"
  fi
  rm -f "${pidfile}"
  # the engine is the server's child; it holds the GPUs until it exits
  local engine_tag="${RUN_DIR}/profiles/${name}-expert-profile.bin"
  local p
  for _ in {1..60}; do
    p="$( { pgrep -f -- "${engine_tag}" || true; } | while read -r q; do if alive "${q}"; then echo "${q}"; fi; done)"
    [[ -n "${p}" ]] || break
    sleep 1
  done
  if [[ -n "${p}" ]]; then
    echo "${name}: engine ${p//$'\n'/ } still running; sending TERM"
    # shellcheck disable=SC2086
    kill ${p} 2>/dev/null || true
  fi
}

stop() {
  stop_one a
  stop_one b
}

wait_url() {
  local name="$1" port="$2"
  local url="http://${HOST}:${port}/v1/models"
  for i in {1..180}; do
    if curl -fsS "${auth_header[@]}" "${url}" >/dev/null 2>&1; then
      echo "${name}: ready (${url})"
      return
    fi
    if ! server_pid "${name}" >/dev/null; then
      tail -80 "${RUN_DIR}/logs/${name}-server.log" >&2 || true
      die "${name}: process exited before readiness"
    fi
    sleep 5
  done
  tail -120 "${RUN_DIR}/logs/${name}-server.log" >&2 || true
  die "${name}: timed out waiting for ${url}"
}

wait_ready() {
  wait_url a "${PORT_A}"
  wait_url b "${PORT_B}"
}

smoke_one() {
  local name="$1" port="$2"
  "${PYTHON}" "${ROOT}/tools/serve_load.py" "http://${HOST}:${port}" \
    --clients 1 --rounds 1 --max-tokens 32 --words 80 \
    "${auth_args[@]}" \
    --json "${RUN_DIR}/results/${name}-smoke.json"
}

smoke() {
  smoke_one a "${PORT_A}"
  smoke_one b "${PORT_B}"
}

load_one() {
  local name="$1" port="$2"
  "${PYTHON}" "${ROOT}/tools/serve_load.py" "http://${HOST}:${port}" \
    --clients "${LOAD_CLIENTS:-1,2,4,8}" \
    --rounds "${LOAD_ROUNDS:-3}" \
    --max-tokens "${LOAD_MAX_TOKENS:-128}" \
    --words "${LOAD_WORDS:-800}" \
    "${auth_args[@]}" \
    --save "${RUN_DIR}/results/${name}-answers" \
    --json "${RUN_DIR}/results/${name}-load.json"
}

load() {
  load_one a "${PORT_A}"
  load_one b "${PORT_B}"
}

long_load_one() {
  local name="$1" port="$2"
  "${PYTHON}" "${ROOT}/tools/serve_longctx_load.py" "http://${HOST}:${port}" \
    --clients "${LONG_CLIENTS:-2,4,8}" \
    --prompt-tokens "${LONG_PROMPT_TOKENS:-32768,64000}" \
    --rounds "${LONG_ROUNDS:-1}" \
    --max-tokens "${LONG_MAX_TOKENS:-64}" \
    "${auth_args[@]}" \
    --save "${RUN_DIR}/results/${name}-long-answers" \
    --json "${RUN_DIR}/results/${name}-long-load.json"
}

long_load() {
  long_load_one a "${PORT_A}"
  long_load_one b "${PORT_B}"
}

status() {
  for name in a b; do
    if server_pid "${name}" >/dev/null; then
      echo "${name}: running pid $(server_pid "${name}")"
    else
      echo "${name}: stopped"
    fi
    if [[ -s "${RUN_DIR}/logs/${name}-server.log" ]]; then
      echo "--- ${name} server log tail ---"
      tail -20 "${RUN_DIR}/logs/${name}-server.log"
    fi
  done
}

case "${ACTION}" in
  prepare) prepare ;;
  start) start ;;
  wait) wait_ready ;;
  smoke) smoke ;;
  load) load ;;
  long-load) long_load ;;
  stop) stop ;;
  status) status ;;
  all)
    prepare
    start
    wait_ready
    smoke
    load
    ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
