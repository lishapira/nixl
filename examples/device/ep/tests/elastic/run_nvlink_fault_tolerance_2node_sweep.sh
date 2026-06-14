#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Two-node NVLink fault-tolerance sweep harness.
#
# Same coverage model as run_nvlink_fault_tolerance_sweep.sh but orchestrates
# across two nodes (8 GPUs, expansion + kill + contraction plan). Delegates
# each run to run_nvlink_fault_tolerance_2node_test.sh.
#
# Environment overrides:
#   SPIN_CYCLES        in-kernel marker spin (default 100000000)
#   ITERATIONS         iterations per timing (default 1)
#   VICTIM_PLAN        default nvlink_fault_tolerance_2node.json
#   BASELINE_PLAN      default nvlink_fault_tolerance_2node_baseline.json
#   RUN_DIR            output dir (default ./results/2node_<utc>_<host>)
#   TIMINGS            space-separated subset (default = all 12)
#   MEM_LEAK_MIB       GPU memory delta tolerance (default 64)
#   SETTLE_SECONDS     post-run settle wait (default 5)

set -uo pipefail

TEST_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "${TEST_DIR}"

LAUNCHER="${TEST_DIR}/run_nvlink_fault_tolerance_2node_test.sh"
[[ -x "${LAUNCHER}" ]] || chmod +x "${LAUNCHER}"

SPIN_CYCLES="${SPIN_CYCLES:-100000000}"
ITERATIONS="${ITERATIONS:-1}"
VICTIM_PLAN="${VICTIM_PLAN:-nvlink_fault_tolerance_2node.json}"
BASELINE_PLAN="${BASELINE_PLAN:-nvlink_fault_tolerance_2node_baseline.json}"
MEM_LEAK_MIB="${MEM_LEAK_MIB:-64}"
SETTLE_SECONDS="${SETTLE_SECONDS:-5}"
export MEM_LEAK_MIB SETTLE_SECONDS

DEFAULT_TIMINGS=(
    before-dispatch
    after-dispatch
    between-dispatch-combine
    before-combine
    after-combine
    dispatch-between-send-receive
    combine-between-send-receive
    dispatch-send-during-kernel-cold
    dispatch-send-during-kernel
    dispatch-receive-during-kernel
    combine-send-during-kernel
    combine-receive-during-kernel
)
if [[ -n "${TIMINGS:-}" ]]; then
    # shellcheck disable=SC2206
    TIMINGS_ARR=( ${TIMINGS} )
else
    TIMINGS_ARR=( "${DEFAULT_TIMINGS[@]}" )
fi

declare -A EXPECTED_TARGET=(
    [dispatch-send-during-kernel]=1
    [dispatch-send-during-kernel-cold]=1
    [dispatch-receive-during-kernel]=2
    [combine-send-during-kernel]=3
    [combine-receive-during-kernel]=4
)

UTC=$(date -u +%Y%m%d_%H%M%S)
HOST=$(hostname -s)
JOB_TAG="${SLURM_JOB_ID:-nojob}"
if [[ -n "${SLURM_NODELIST:-}" ]]; then
    NODES_STR=$(scontrol show hostnames "${SLURM_NODELIST}" | paste -sd_)
else
    NODES_STR="manual"
fi
RUN_DIR="${RUN_DIR:-${TEST_DIR}/results/2node_${UTC}_job${JOB_TAG}_${NODES_STR}}"
mkdir -p "${RUN_DIR}"
SUMMARY="${RUN_DIR}/SUMMARY.md"
EVIDENCE_DIR="${RUN_DIR}/evidence"
mkdir -p "${EVIDENCE_DIR}"

log_summary() { printf '%s\n' "$*" >> "${SUMMARY}"; }

parse_cleanup_from_log() {
    local log="$1"
    local result_line
    result_line=$(grep -E '^cleanup_result:' "${log}" 2>/dev/null | head -n 1 || true)
    if [[ -z "${result_line}" ]]; then
        CLEANUP_STATUS="UNKNOWN"
        CLEANUP_DETAIL="no CLEANUP REPORT in log"
        return
    fi
    local delta_line delta
    delta_line=$(grep -E '^gpu_mem_used_mib:' "${log}" 2>/dev/null | head -n 1 || true)
    delta=$(printf '%s' "${delta_line}" | sed -n 's/.*delta=\(-\?[0-9]\+\).*/\1/p')
    if [[ "${result_line}" == *CLEAN* ]]; then
        CLEANUP_STATUS="PASS"
        CLEANUP_DETAIL=$(printf 'delta=%+dMiB' "${delta:-0}")
    else
        CLEANUP_STATUS="FAIL"
        local detail
        detail=$(printf '%s' "${result_line}" \
            | sed -n 's/^cleanup_result:[[:space:]]*DIRTY[[:space:]]*(\(.*\))$/\1/p')
        CLEANUP_DETAIL="${detail:-DIRTY}"
    fi
}

verify_build() {
    local probe_log="${RUN_DIR}/build_probe.log"
    local py_out probe_node="${MASTER_NODE:-}"
    if [[ -z "${probe_node}" && -n "${SLURM_NODELIST:-${SLURM_JOB_NODELIST:-}}" ]]; then
        if command -v scontrol >/dev/null 2>&1; then
            probe_node=$(scontrol show hostnames "${SLURM_NODELIST:-${SLURM_JOB_NODELIST}}" | head -n 1)
        fi
    fi
    local probe_py='source /workspace/lishapira/setup_node.sh >/dev/null 2>&1; python3 - <<'\''PY'\''
import nixl_ep
needed = (
    "enable_in_kernel_fault_marker",
    "disable_in_kernel_fault_marker",
    "get_in_kernel_fault_marker_snapshot",
    "query_mask_buffer",
)
missing = [m for m in needed if m not in dir(nixl_ep.Buffer)]
if missing:
    raise SystemExit("missing Buffer methods: " + ", ".join(missing))
print("Buffer methods OK")
PY'
    if command -v srun >/dev/null 2>&1 && [[ -n "${probe_node}" ]] && [[ -n "${SLURM_JOB_ID:-}" ]]; then
        local srun_probe=(srun --overlap --nodes=1 --nodelist="${probe_node}")
        [[ -n "${CONTAINER_IMAGE:-}" ]] && srun_probe+=(--container-image="${CONTAINER_IMAGE}")
        [[ -n "${CONTAINER_MOUNTS:-}" ]] && srun_probe+=(--container-mounts="${CONTAINER_MOUNTS}" --container-workdir=/workspace/lishapira)
        py_out=$("${srun_probe[@]}" bash -lc "${probe_py}" 2>&1)
        {
            echo "=== python import probe (via srun on ${probe_node}) ==="
            echo "${py_out}"
        } > "${probe_log}"
    else
        local install_dir="${NIXL_PREFIX:-${NIXL_INSTALL:-/workspace/lishapira/nixl/install}}"
        py_out=$(
            PYTHONPATH="${install_dir}/lib/python3/dist-packages:${PYTHONPATH:-}" \
            LD_LIBRARY_PATH="${install_dir}/lib/${ARCH:-$(uname -m)}-linux-gnu:${install_dir}/lib:${LD_LIBRARY_PATH:-}" \
            python3 - <<'PY' 2>&1
import nixl_ep
needed = (
    "enable_in_kernel_fault_marker",
    "disable_in_kernel_fault_marker",
    "get_in_kernel_fault_marker_snapshot",
    "query_mask_buffer",
)
missing = [m for m in needed if m not in dir(nixl_ep.Buffer)]
if missing:
    raise SystemExit("missing Buffer methods: " + ", ".join(missing))
print("Buffer methods OK")
PY
        )
        {
            echo "=== python import probe (local) ==="
            echo "${py_out}"
        } > "${probe_log}"
    fi
    if ! grep -q "Buffer methods OK" "${probe_log}"; then
        echo "BUILD SANITY FAILED -- rebuild from nvlink_fault_tolerance branch." >&2
        echo "See ${probe_log}" >&2
        return 1
    fi
    return 0
}

expected_survivors_for_plan() {
    local plan="$1"
    python3 - "$plan" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
last = plan[-1]
print(sum(1 for r in last if r >= 0))
PY
}

run_one() {
    local plan="$1"; shift
    local log="$1"; shift
    PLAN_FILE="${plan}" \
    FAULT_EVIDENCE_DIR="${EVIDENCE_DIR}" \
        bash "${LAUNCHER}" "$@" > "${log}" 2>&1
}

analyse_run() {
    local timing="$1"
    local log="$2"
    local fault_rc="$3"
    local expected_survivors="$4"
    local cleanup_status="$5"
    local cleanup_detail="$6"

    local hit="N/A"
    local target_match="N/A"
    local exp_target="${EXPECTED_TARGET[${timing}]:-}"
    if [[ -n "${exp_target}" ]]; then
        local evidence_glob="${EVIDENCE_DIR}/in_kernel_fault_rank*_${timing}_pid*.log"
        # shellcheck disable=SC2086
        local evidence_files=( ${evidence_glob} )
        if [[ -e "${evidence_files[0]}" ]]; then
            if grep -q HIT_IN_KERNEL_WINDOW "${evidence_files[@]}"; then
                hit="HIT"
            elif grep -q MISSED_IN_KERNEL_TIMING "${evidence_files[@]}"; then
                hit="MISSED"
            elif grep -q IN_KERNEL_MARKER_TIMEOUT "${evidence_files[@]}"; then
                hit="TIMEOUT"
            else
                hit="NO_VERDICT"
            fi
            local got_target
            got_target=$(sed -n 's/^target=//p' "${evidence_files[@]}" | head -n 1)
            if [[ "${got_target}" == "${exp_target}" ]]; then
                target_match="ok(${got_target})"
            else
                target_match="MISMATCH(want=${exp_target},got=${got_target:-none})"
            fi
        else
            hit="NO_EVIDENCE"
        fi
    fi

    local done_count
    done_count=$(grep -c '\[rank [0-9]\+\] WORKER DONE survived=true' "${log}" || true)
    local survivors_field="${done_count}/${expected_survivors}"

    local tb_count
    tb_count=$(grep -E -c 'Traceback|AssertionError|RuntimeError' "${log}" || true)

    local mask_pass=0 mask_total=0
    while IFS= read -r line; do
        local p c
        p=$(printf '%s' "${line}" | sed -n 's/.*passes=\([0-9]\+\).*/\1/p')
        c=$(printf '%s' "${line}" | sed -n 's/.*calls=\([0-9]\+\).*/\1/p')
        if [[ -n "${p}" && -n "${c}" ]]; then
            mask_pass=$((mask_pass + p))
            mask_total=$((mask_total + c))
        fi
    done < <(grep '\[rank [0-9]\+\] MASK CHECK SUMMARY' "${log}" || true)
    local mask_field
    if [[ "${mask_total}" -eq 0 ]]; then
        mask_field="n/a"
    elif [[ "${mask_pass}" -eq "${mask_total}" ]]; then
        mask_field="${mask_pass}/${mask_total}"
    else
        mask_field="${mask_pass}/${mask_total}!"
    fi

    local detect_ms="n/a"
    local kill_ns=""
    if [[ -n "${exp_target}" ]]; then
        local evidence_glob_for_ts="${EVIDENCE_DIR}/in_kernel_fault_rank*_${timing}_pid*.log"
        # shellcheck disable=SC2086
        local evidence_for_ts=( ${evidence_glob_for_ts} )
        if [[ -e "${evidence_for_ts[0]}" ]]; then
            kill_ns=$(grep -h '^timestamp_ns=' "${evidence_for_ts[@]}" 2>/dev/null \
                | sed 's/^timestamp_ns=//' | head -n 1)
        fi
    else
        kill_ns=$(grep -E '\[rank [0-9]+\] Killing rank at ' "${log}" \
            | sed -n 's/.*timestamp_ns=\([0-9]\+\).*/\1/p' | head -n 1)
    fi
    if [[ -n "${kill_ns}" ]]; then
        local det_ns
        det_ns=$(grep '\[rank [0-9]\+\] MASK DETECTED ' "${log}" 2>/dev/null \
            | sed -n 's/.*timestamp_ns=\([0-9]\+\).*/\1/p' \
            | awk -v k="${kill_ns}" '$1+0 >= k+0' \
            | sort -n | head -n 1)
        if [[ -n "${det_ns}" ]]; then
            detect_ms=$(awk -v a="${kill_ns}" -v b="${det_ns}" 'BEGIN{printf "%.1f", (b-a)/1e6}')
        fi
    fi

    local verdict="PASS"
    local fail_reasons=()
    [[ "${fault_rc}" -eq 0 ]] || fail_reasons+=("fault_rc=${fault_rc}")
    [[ "${done_count}" -eq "${expected_survivors}" ]] || fail_reasons+=("survivors=${done_count}/${expected_survivors}")
    [[ "${tb_count}" -eq 0 ]] || fail_reasons+=("tracebacks=${tb_count}")
    if [[ "${mask_total}" -gt 0 && "${mask_pass}" -ne "${mask_total}" ]]; then
        fail_reasons+=("mask=${mask_pass}/${mask_total}")
    fi
    if [[ -n "${exp_target}" ]]; then
        [[ "${hit}" == "HIT" ]] || fail_reasons+=("hit=${hit}")
        [[ "${target_match}" == ok* ]] || fail_reasons+=("target=${target_match}")
    fi
    [[ "${cleanup_status}" == "PASS" ]] || fail_reasons+=("cleanup=${cleanup_detail}")
    if [[ "${#fail_reasons[@]}" -gt 0 ]]; then
        verdict="FAIL(${fail_reasons[*]})"
    fi

    ROW_FAULT_RC="${fault_rc}"
    ROW_HIT="${hit}"
    ROW_TARGET="${target_match}"
    ROW_SURVIVORS="${survivors_field}"
    ROW_MASK="${mask_field}"
    ROW_LATENCY="${detect_ms}"
    ROW_TB="${tb_count}"
    ROW_CLEANUP="${cleanup_detail}"
    ROW_VERDICT="${verdict}"
}

{
    echo "# Two-node NVLink fault-tolerance sweep"
    echo
    echo "- Started UTC: \`${UTC}\`"
    echo "- Job: \`${JOB_TAG}\`"
    echo "- Nodes: \`${NODES_STR}\`"
    echo "- Victim plan: \`${VICTIM_PLAN}\`"
    echo "- Baseline plan: \`${BASELINE_PLAN}\`"
    echo "- Run dir: \`${RUN_DIR}\`"
    echo
} > "${SUMMARY}"

echo "[2node-sweep] verifying build..." >&2
if ! verify_build; then
    log_summary "Build sanity FAILED. See \`build_probe.log\`."
    exit 1
fi
log_summary "- Build sanity: PASS"

EXPECTED_SURVIVORS_FAULT=$(expected_survivors_for_plan "${VICTIM_PLAN}")
EXPECTED_SURVIVORS_BASE=$(expected_survivors_for_plan "${BASELINE_PLAN}")
log_summary "- Expected survivors (fault / baseline): \`${EXPECTED_SURVIVORS_FAULT}\` / \`${EXPECTED_SURVIVORS_BASE}\`"

init_log="${RUN_DIR}/initial_baseline.log"
run_one "${BASELINE_PLAN}" "${init_log}"
init_rc=$?
init_done=$(grep -c '\[rank [0-9]\+\] WORKER DONE survived=true' "${init_log}" || true)
if [[ "${init_rc}" -ne 0 || "${init_done}" -ne "${EXPECTED_SURVIVORS_BASE}" ]]; then
    log_summary "- Initial baseline: FAIL (rc=${init_rc}, survivors=${init_done}/${EXPECTED_SURVIVORS_BASE})"
    exit 1
fi
log_summary "- Initial baseline: PASS (rc=0, survivors=${init_done}/${EXPECTED_SURVIVORS_BASE})"

log_summary ""
log_summary "## Per-timing results"

OVERALL=0

for timing in "${TIMINGS_ARR[@]}"; do
    for iter in $(seq 1 "${ITERATIONS}"); do
        echo "[2node-sweep] ${timing} iter=${iter}/${ITERATIONS}" >&2
        rm -f "${EVIDENCE_DIR}"/in_kernel_fault_rank*_"${timing}"_pid*.log

        log="${RUN_DIR}/${timing}__iter${iter}.log"
        is_in_kernel=0
        if [[ -n "${EXPECTED_TARGET[${timing}]:-}" ]]; then
            is_in_kernel=1
            run_one "${VICTIM_PLAN}" "${log}" \
                --fault-kill-timing "${timing}" \
                --in-kernel-fault-spin-cycles "${SPIN_CYCLES}"
        else
            run_one "${VICTIM_PLAN}" "${log}" \
                --fault-kill-timing "${timing}"
        fi
        fault_rc=$?

        parse_cleanup_from_log "${log}"
        analyse_run "${timing}" "${log}" "${fault_rc}" \
            "${EXPECTED_SURVIVORS_FAULT}" "${CLEANUP_STATUS}" "${CLEANUP_DETAIL}"

        post_log="${RUN_DIR}/${timing}__iter${iter}__post_baseline.log"
        run_one "${BASELINE_PLAN}" "${post_log}"
        post_rc=$?
        post_done=$(grep -c '\[rank [0-9]\+\] WORKER DONE survived=true' "${post_log}" || true)
        post_field="rc=${post_rc} survivors=${post_done}/${EXPECTED_SURVIVORS_BASE}"
        if [[ "${post_rc}" -ne 0 || "${post_done}" -ne "${EXPECTED_SURVIVORS_BASE}" ]]; then
            ROW_VERDICT="FAIL(post_baseline=${post_field})"
        fi

        log_summary ""
        log_summary "### \`${timing}\` (run ${iter})"
        log_summary "- Fault run exit code: \`${ROW_FAULT_RC}\`"
        if (( is_in_kernel == 1 )); then
            log_summary "- In-kernel spin cycles: \`${SPIN_CYCLES}\`"
            log_summary "- Marker HIT verdict: ${ROW_HIT}"
            log_summary "- In-kernel target slot: ${ROW_TARGET}"
        fi
        log_summary "- Survivors (observed/expected): ${ROW_SURVIVORS}"
        log_summary "- Mask-check passes: ${ROW_MASK}"
        log_summary "- Mask propagation latency: ${ROW_LATENCY} ms"
        log_summary "- Survivor tracebacks: ${ROW_TB}"
        log_summary "- GPU memory delta: ${ROW_CLEANUP}"
        log_summary "- Post-fault baseline: ${post_field}"
        log_summary "- Verdict: ${ROW_VERDICT}"

        [[ "${ROW_VERDICT}" != PASS* ]] && OVERALL=1
    done
done

log_summary ""
log_summary "## Final"
if [[ "${OVERALL}" -eq 0 ]]; then
    log_summary "- Overall: **PASS**"
else
    log_summary "- Overall: **FAIL**"
fi

echo "[2node-sweep] done; see ${SUMMARY}" >&2
exit "${OVERALL}"
