#!/usr/bin/env bash
#SBATCH --account=network_research_advdev
#SBATCH --partition=gb300-backfill
#SBATCH --job-name=network_research_advdev-nixl.ep.phase3-build
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fsw/network_research_advdev/lishapira/nixl/nixl_ep_nvlink_understanding/common/legacy_helpers/phase3-build-%j.out

set -euo pipefail

ROOT=/lustre/fsw/network_research_advdev/lishapira
IMAGE=$ROOT/nixl-hybrid-ep-cuda2.sqsh
MOUNTS="$ROOT:/workspace/lishapira"

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "node_list=${SLURM_JOB_NODELIST:-unknown}"
echo "image=$IMAGE"

srun -N1 -n1 \
  --container-image="$IMAGE" \
  --container-mounts="$MOUNTS" \
  --container-workdir=/workspace/lishapira \
  bash -lc '
    set -euo pipefail
    cd /workspace/lishapira/nixl
    git status --short
    /workspace/lishapira/build_nixl_aarch64.sh
  '
