#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Single-node NVLink fault-tolerance sweep harness.
#
# Runs the elastic EP test once per fault-kill-timing, and for each run
# verifies properties beyond "the process exited 0":
#
#   - Fault rc                  -- elastic.py exit code (0 on a passing fault run)
#   - In-kernel hit verdict     -- HIT_IN_KERNEL_WINDOW / MISSED / TIMEOUT
#   - Evidence target matches   -- the GPU marker fired in the requested phase
#   - Survivors completed       -- count of "WORKER DONE" lines == expected survivors
#   - No tracebacks             -- elastic.py log has no Traceback / AssertionError
#   - Mask check pass           -- every survivor printed a MASK CHECK SUMMARY
#                                  with passes == calls
#   - Detection latency         -- ms between killed rank's "Killing rank ..."
#                                  and survivor's "detected unexpected rank failures"
#   - Cleanup: no leftover procs (without pkill), no stuck nvidia-smi compute
#     apps, GPU memory delta back near baseline, no leaked TCPStore/rank-server
#     ports, no /dev/shm/torch_* fragments
#   - Post-baseline rc          -- the no-fault baseline still passes after the
#                                  fault run cleaned up
#
# All per-run logs land in $RUN_DIR. A SUMMARY.md table at the end lists one
# row per timing.
#
# This script orchestrates; it delegates each elastic.py invocation to
# run_nvlink_fault_tolerance_test.sh (the existing single-run launcher), so
# everything we set up there (PYTHONPATH, LD_LIBRARY_PATH, unset UCX_TLS,
# default SIGKILL) applies automatically.
#
# Environment overrides:
#   SPIN_CYCLES        in-kernel marker spin (default 100000000 == ~50ms on H100)
#   ITERATIONS         iterations per timing (default 1)
#   VICTIM_PLAN        plan with the victim rank (default nvlink_fault_tolerance.json)
#   BASELINE_PLAN      no-kill plan (default nvlink_fault_tolerance_baseline.json)
#   RUN_DIR            output dir (default ./results/<utc>_<host>)
#   TIMINGS            space-separated subset to run (default = all 12)
#   MEM_LEAK_MIB       GPU memory delta tolerance in MiB (default 64)
#   SETTLE_SECONDS     seconds to wait after elastic exits before cleanup probe (default 5)

set -uo pipefail

TEST_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "${TEST_DIR}"

LAUNCHER="${TEST_DIR}/run_nvlink_fault_tolerance_test.sh"
[[ -x "${LAUNCHER}" ]] || { echo "missing executable launcher: ${LAUNCHER}" >&2; exit 2; }

# ---------------------------------------------------------------------------
# Defaults / overrides
# ---------------------------------------------------------------------------
SPIN_CYCLES="${SPIN_CYCLES:-100000000}"
ITERATIONS="${ITERATIONS:-1}"
VICTIM_PLAN="${VICTIM_PLAN:-nvlink_fault_tolerance.json}"
BASELINE_PLAN="${BASELINE_PLAN:-nvlink_fault_tolerance_baseline.json}"
MEM_LEAK_MIB="${MEM_LEAK_MIB:-64}"
SETTLE_SECONDS="${SETTLE_SECONDS:-5}"

DEFAULT_TIMINGS=(
    # CPU-level kills (host signals SIGKILL between ops)
    before-dispatch
    after-dispatch
    between-dispatch-combine
    before-combine
    after-combine
    # CPU-level kills inside the host-visible send/receive seam (hook pass only)
    dispatch-between-send-receive
    combine-between-send-receive
    # In-kernel kills (GPU marker -> host SIGKILL while kernel is in the phase)
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

UTC=$(date -u +%Y%m%d_%H%M%S)
HOST=$(hostname -s)
RUN_DIR="${RUN_DIR:-${TEST_DIR}/results/${UTC}_${HOST}}"
mkdir -p "${RUN_DIR}"
SUMMARY="${RUN_DIR}/SUMMARY.md"
EVIDENCE_DIR="${RUN_DIR}/evidence"
mkdir -p "${EVIDENCE_DIR}"

# Each timing's expected in-kernel marker target id; CPU-level timings have ""
declare -A EXPECTED_TARGET=(
    [dispatch-send-during-kernel]=1
    [dispatch-send-during-kernel-cold]=1
    [dispatch-receive-during-kernel]=2
    [combine-send-during-kernel]=3
    [combine-receive-during-kernel]=4
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log_summary() { printf '%s\n' "$*" >> "${SUMMARY}"; }

# Sum GPU memory.used across all visible GPUs, in MiB.
gpu_mem_used_mib() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
        | awk 'BEGIN{s=0} {s+=$1} END{print s}'
}

# Number of leftover processes that look like elastic / rank server / torch
# multiprocessing workers. NO pkill -- we want the test's own cleanup to be
# what's being measured.
leftover_proc_count() {
    pgrep -af 'elastic\.py|rank_server|spawn_main|torch.multiprocessing' 2>/dev/null \
        | wc -l
}

# /dev/shm leak count from torch multiprocessing.
shm_leak_count() {
    ls /dev/shm/torch_* /dev/shm/cuda.shm.* 2>/dev/null | wc -l
}

# Does anything still hold the rank server or TCPStore port?
ports_in_use() {
    local count=0
    if command -v ss >/dev/null 2>&1; then
        count=$(ss -tlnH 2>/dev/null | awk '$4 ~ /:(9999|10000)$/' | wc -l)
    elif command -v netstat >/dev/null 2>&1; then
        count=$(netstat -tlnp 2>/dev/null | awk '$4 ~ /:(9999|10000)$/' | wc -l)
    fi
    echo "$count"
}

# nvidia-smi --query-compute-apps lines (non-empty == something is holding a
# GPU context).
nvsmi_compute_app_count() {
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
        | awk 'NF>0' | wc -l
}

# Pre-flight: verify the install actually has Lior's in-kernel marker symbols.
#
# The authoritative check is "do the new pybind methods show up on
# `nixl_ep.Buffer`?". The methods only exist when the bindings + C++ marker
# code were compiled in, so this is equivalent to a `strings | grep
# in_kernel_fault_marker` check but doesn't depend on knowing the exact
# install-tree path of the compiled .so. We additionally cross-check by
# running `strings` on the .so file Python actually loaded (path discovered
# via importlib), but treat that as informational only -- it is intentionally
# not used as a failure gate so install-layout differences across builds do
# not produce false BUILD SANITY failures.
verify_build() {
    local install_dir="${NIXL_INSTALL:-/workspace/nixl/install}"
    local probe_log="${RUN_DIR}/build_probe.log"
    {
        echo "=== install dir ==="
        ls -ld "${install_dir}" 2>&1 || true
    } > "${probe_log}"
    local py_out
    py_out=$(
        PYTHONPATH="${install_dir}/lib/python3/dist-packages:${PYTHONPATH:-}" \
        LD_LIBRARY_PATH="${install_dir}/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}" \
        python3 - <<'PY' 2>&1
import nixl_ep, importlib.util
print("nixl_ep package file:", nixl_ep.__file__)
spec = importlib.util.find_spec("nixl_ep.nixl_ep_cpp")
print("nixl_ep_cpp .so file:", spec.origin if spec else "(unresolved)")
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
        echo "=== python import probe ==="
        echo "${py_out}"
        echo "=== informational strings probe ==="
        local so_path
        so_path=$(printf '%s\n' "${py_out}" \
            | sed -n 's|^nixl_ep_cpp \.so file: ||p' \
            | head -n 1)
        if [[ -n "${so_path}" && "${so_path}" != "(unresolved)" && -f "${so_path}" ]]; then
            local count
            count=$(strings "${so_path}" 2>/dev/null | grep -c in_kernel_fault_marker || true)
            echo "in_kernel_fault_marker symbols in ${so_path}: ${count:-0}"
        else
            echo "(skipped: could not resolve loaded .so path)"
        fi
    } >> "${probe_log}"
    if ! grep -q "Buffer methods OK" "${probe_log}"; then
        echo "BUILD SANITY FAILED -- the install's nixl_ep.Buffer is missing in-kernel marker methods." >&2
        echo "See ${probe_log}" >&2
        return 1
    fi
    return 0
}

# Compute expected survivor count from a plan: ranks present and non-negative
# in the last phase. Uses python to keep us robust against multi-phase plans.
expected_survivors_for_plan() {
    local plan="$1"
    python3 - "$plan" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
last = plan[-1]
print(sum(1 for r in last if r >= 0))
PY
}

# Single elastic run via the underlying launcher. Args after the plan are
# forwarded verbatim. Captures stdout+stderr to the given log.
run_one() {
    local plan="$1"; shift
    local log="$1"; shift
    PLAN_FILE="${plan}" \
    FAULT_EVIDENCE_DIR="${EVIDENCE_DIR}" \
        bash "${LAUNCHER}" "$@" > "${log}" 2>&1
}

# Parse one fault run's log + evidence and return PASS/FAIL row text.
# Sets globals: ROW_RESULT, ROW_DETAIL.
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
    done_count=$(grep -c '^\[rank [0-9]\+\] WORKER DONE survived=true' "${log}" || true)
    local survivors_field="${done_count}/${expected_survivors}"

    local tb_count
    tb_count=$(grep -E -c 'Traceback|AssertionError|RuntimeError' "${log}" || true)
    # Suppress the expected RuntimeError that elastic.py raises only when a
    # worker has an unexpected exit code; on a clean fault run nothing matches.

    local mask_pass=0 mask_total=0
    while IFS= read -r line; do
        local p c
        p=$(printf '%s' "${line}" | sed -n 's/.*passes=\([0-9]\+\).*/\1/p')
        c=$(printf '%s' "${line}" | sed -n 's/.*calls=\([0-9]\+\).*/\1/p')
        if [[ -n "${p}" && -n "${c}" ]]; then
            mask_pass=$((mask_pass + p))
            mask_total=$((mask_total + c))
        fi
    done < <(grep '^\[rank [0-9]\+\] MASK CHECK SUMMARY' "${log}" || true)
    local mask_field
    if [[ "${mask_total}" -eq 0 ]]; then
        mask_field="n/a"
    elif [[ "${mask_pass}" -eq "${mask_total}" ]]; then
        mask_field="${mask_pass}/${mask_total}"
    else
        mask_field="${mask_pass}/${mask_total}!"
    fi

    # Mask-propagation latency: ns delta between the victim's kill timestamp
    # (either `Killing rank at ... timestamp_ns=` for CPU-level timings, or
    # `HIT_IN_KERNEL_WINDOW ... timestamp_ns=` for in-kernel timings -- both
    # are emitted on the victim's stdout from elastic.py) and the FIRST
    # `MASK DETECTED ... timestamp_ns=` line on any survivor, where survivors
    # emit one such line the first time each peer transitions from alive to
    # dead in their runtime mask.
    #
    # The OLD metric grepped for `detected unexpected rank failures`, which
    # only fires at the end of test_main in worker(), so it was dominated by
    # the survivor's remaining iteration budget rather than mask propagation.
    # In-kernel timings also produced `n/a` under the old metric because the
    # in-kernel kill path never emits a survivor-side end-of-phase line in
    # time -- the survivor only learns at the next phase boundary.
    local detect_ms="n/a"
    if [[ -n "${exp_target}" || "$(grep -c 'Killing rank at' "${log}" || true)" -gt 0 ]]; then
        local kill_ns
        kill_ns=$(grep -E '^\[rank [0-9]+\] (Killing rank at|HIT_IN_KERNEL_WINDOW) ' "${log}" \
            | sed -n 's/.*timestamp_ns=\([0-9]\+\).*/\1/p' | head -n 1)
        local det_ns
        # Take the EARLIEST MASK DETECTED timestamp on any survivor that is
        # at or after the victim's kill timestamp. Multiple survivors will
        # each emit their own; we want the fastest observer.
        if [[ -n "${kill_ns}" ]]; then
            det_ns=$(grep '^\[rank [0-9]\+\] MASK DETECTED ' "${log}" \
                | sed -n 's/.*timestamp_ns=\([0-9]\+\).*/\1/p' \
                | awk -v k="${kill_ns}" '$1+0 >= k+0' \
                | sort -n | head -n 1)
        fi
        if [[ -n "${kill_ns}" && -n "${det_ns}" ]]; then
            detect_ms=$(awk -v a="${kill_ns}" -v b="${det_ns}" 'BEGIN{printf "%.1f", (b-a)/1e6}')
        fi
    fi

    # Verdict
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

    ROW_RESULT="${verdict}"
    ROW_DETAIL=$(printf '%s | %s | %s | %s | %s | %s ms | %s | %s | %s' \
        "${fault_rc}" "${hit}" "${target_match}" "${survivors_field}" \
        "${mask_field}" "${detect_ms}" "${tb_count}" "${cleanup_detail}" \
        "${verdict}")
}

# Capture and compare per-timing cleanup state. Returns status via globals
# CLEANUP_STATUS and CLEANUP_DETAIL.
check_cleanup() {
    local baseline_mem="$1"
    local timing="$2"
    sleep "${SETTLE_SECONDS}"
    local procs shm ports nvsmi_apps mem_after delta
    procs=$(leftover_proc_count)
    shm=$(shm_leak_count)
    ports=$(ports_in_use)
    nvsmi_apps=$(nvsmi_compute_app_count)
    mem_after=$(gpu_mem_used_mib)
    delta=$((mem_after - baseline_mem))
    local issues=()
    [[ "${procs}" -eq 0 ]]      || issues+=("procs=${procs}")
    [[ "${shm}" -eq 0 ]]        || issues+=("shm=${shm}")
    [[ "${ports}" -eq 0 ]]      || issues+=("ports=${ports}")
    [[ "${nvsmi_apps}" -eq 0 ]] || issues+=("nvsmi_apps=${nvsmi_apps}")
    if [[ "${delta}" -gt "${MEM_LEAK_MIB}" ]]; then
        issues+=("mem_delta=+${delta}MiB")
    fi
    if [[ "${#issues[@]}" -eq 0 ]]; then
        CLEANUP_STATUS="PASS"
        CLEANUP_DETAIL=$(printf 'delta=+%dMiB' "${delta}")
    else
        CLEANUP_STATUS="FAIL"
        CLEANUP_DETAIL="${issues[*]}"
    fi
}

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------

{
    echo "# NVLink fault-tolerance sweep"
    echo
    echo "- Started UTC: \`${UTC}\`"
    echo "- Host: \`${HOST}\`"
    echo "- Spin cycles: \`${SPIN_CYCLES}\`"
    echo "- Iterations per timing: \`${ITERATIONS}\`"
    echo "- Victim plan: \`${VICTIM_PLAN}\`"
    echo "- Baseline plan: \`${BASELINE_PLAN}\`"
    echo "- Mem leak tolerance: \`${MEM_LEAK_MIB}\` MiB"
    echo "- Settle seconds: \`${SETTLE_SECONDS}\`"
    echo "- Run dir: \`${RUN_DIR}\`"
    echo
} > "${SUMMARY}"

echo "[sweep] verifying build..." >&2
if ! verify_build; then
    log_summary "Build sanity FAILED. See \`build_probe.log\`."
    exit 1
fi
log_summary "- Build sanity: PASS"

# Establish memory baseline AFTER build probe (which doesn't allocate GPU mem).
BASELINE_MEM=$(gpu_mem_used_mib)
log_summary "- Baseline GPU memory used: \`${BASELINE_MEM}\` MiB"

EXPECTED_SURVIVORS_FAULT=$(expected_survivors_for_plan "${VICTIM_PLAN}")
EXPECTED_SURVIVORS_BASE=$(expected_survivors_for_plan "${BASELINE_PLAN}")
log_summary "- Expected survivors (fault plan / baseline plan): \`${EXPECTED_SURVIVORS_FAULT}\` / \`${EXPECTED_SURVIVORS_BASE}\`"

echo "[sweep] initial baseline..." >&2
init_log="${RUN_DIR}/initial_baseline.log"
run_one "${BASELINE_PLAN}" "${init_log}"
init_rc=$?
init_done=$(grep -c '^\[rank [0-9]\+\] WORKER DONE survived=true' "${init_log}" || true)
if [[ "${init_rc}" -ne 0 || "${init_done}" -ne "${EXPECTED_SURVIVORS_BASE}" ]]; then
    log_summary "- Initial baseline: FAIL (rc=${init_rc}, survivors=${init_done}/${EXPECTED_SURVIVORS_BASE}). See \`initial_baseline.log\`."
    exit 1
fi
log_summary "- Initial baseline: PASS (rc=0, survivors=${init_done}/${EXPECTED_SURVIVORS_BASE})"

log_summary ""
log_summary "## Per-timing results"
log_summary ""
log_summary "| Fault timing | Run # | In-kernel spin cycles | Fault run exit code | Marker HIT verdict | In-kernel target slot | Survivors (observed/expected) | Mask-check passes (passes/total) | Mask propagation latency (ms) | Survivor tracebacks | GPU memory delta vs baseline | Verdict (incl. post-fault baseline rerun) |"
log_summary "|---|---:|---:|---:|---|---|---|---|---:|---:|---|---|"

OVERALL=0

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
for timing in "${TIMINGS_ARR[@]}"; do
    for iter in $(seq 1 "${ITERATIONS}"); do
        echo "[sweep] ${timing} iter=${iter}/${ITERATIONS}" >&2
        # Wipe any prior evidence for this timing so we read fresh state.
        rm -f "${EVIDENCE_DIR}"/in_kernel_fault_rank*_"${timing}"_pid*.log

        log="${RUN_DIR}/${timing}__iter${iter}.log"
        # In-kernel timings need the spin knob; CPU-level timings ignore it
        # but we don't forward it (keeps the cmd line clean in logs).
        if [[ -n "${EXPECTED_TARGET[${timing}]:-}" ]]; then
            run_one "${VICTIM_PLAN}" "${log}" \
                --fault-kill-timing "${timing}" \
                --in-kernel-fault-spin-cycles "${SPIN_CYCLES}"
            spin_field="${SPIN_CYCLES}"
        else
            run_one "${VICTIM_PLAN}" "${log}" \
                --fault-kill-timing "${timing}"
            spin_field="n/a"
        fi
        fault_rc=$?

        check_cleanup "${BASELINE_MEM}" "${timing}"

        analyse_run "${timing}" "${log}" "${fault_rc}" \
            "${EXPECTED_SURVIVORS_FAULT}" "${CLEANUP_STATUS}" "${CLEANUP_DETAIL}"

        post_log="${RUN_DIR}/${timing}__iter${iter}__post_baseline.log"
        run_one "${BASELINE_PLAN}" "${post_log}"
        post_rc=$?
        post_done=$(grep -c '^\[rank [0-9]\+\] WORKER DONE survived=true' "${post_log}" || true)
        post_field="rc=${post_rc} survivors=${post_done}/${EXPECTED_SURVIVORS_BASE}"
        if [[ "${post_rc}" -ne 0 || "${post_done}" -ne "${EXPECTED_SURVIVORS_BASE}" ]]; then
            ROW_RESULT="FAIL(post_baseline=${post_field})"
        fi

        log_summary "| \`${timing}\` | ${iter} | \`${spin_field}\` | ${ROW_DETAIL} (post: ${post_field}) |"
        [[ "${ROW_RESULT}" != PASS* ]] && OVERALL=1
    done
done

# ---------------------------------------------------------------------------
# Final
# ---------------------------------------------------------------------------
log_summary ""
log_summary "## Final"
log_summary ""
if [[ "${OVERALL}" -eq 0 ]]; then
    log_summary "- Overall: **PASS**"
else
    log_summary "- Overall: **FAIL** (one or more rows failed; grep \`FAIL\` above)"
fi
log_summary "- Finished UTC: \`$(date -u +%Y%m%d_%H%M%S)\`"

echo "[sweep] done; see ${SUMMARY}" >&2
exit "${OVERALL}"
