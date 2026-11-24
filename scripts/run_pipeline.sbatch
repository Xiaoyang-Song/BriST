#!/bin/bash
# Full BriST pipeline on both simulations. Every step skips itself if its output already exists.
#   sbatch scripts/run_pipeline.sbatch
# Edit --account / --partition for your allocation. Takes about 6 h on one V100.
#SBATCH --job-name=brist
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=runs/slurm-%j.log

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/nfs/turbo/coe-sunwbgt/xysong/BriST}"
export HF_HOME=$PWD/hf_cache PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false
PY=$PWD/.venv/bin/python
mkdir -p runs runs_dcca

students () {  # $1 = output root, $2 = dataset
  R=$1; D=$2; S="$PY scripts/stage3_student.py --root $R --data $D"
  for m in feature token hier selective; do
    [ -f $R/align_$m/align.pt ] || $PY scripts/stage2_align.py --root $R --data $D --mode $m > $R/q_align_$m.log 2>&1
  done
  [ -f $R/student/direct/metrics.json ]         || $S --name direct --decoder base > $R/q_direct.log 2>&1
  [ -f $R/student/token_zeroshot/metrics.json ] || $S --name token_zeroshot --align $R/align_token --zero-shot > $R/q_token_zeroshot.log 2>&1
  [ -f $R/student/feature_vlm/metrics.json ]    || $S --name feature_vlm --align $R/align_feature > $R/q_feature_vlm.log 2>&1
  [ -f $R/student/hier_vlm/metrics.json ]       || $S --name hier_vlm --align $R/align_hier > $R/q_hier_vlm.log 2>&1
  [ -f $R/student/hier_kd/metrics.json ]        || $S --name hier_kd --align $R/align_hier --kd 1.0 --tok 0.1 > $R/q_hier_kd.log 2>&1
  [ -f $R/student/selective_kd/metrics.json ]   || $S --name selective_kd --align $R/align_selective --kd 1.0 --tok 0.1 > $R/q_selective_kd.log 2>&1
  $PY scripts/monitor.py --root $R --data $D --students direct token_zeroshot feature_vlm hier_vlm hier_kd selective_kd > $R/q_monitor.log 2>&1
  $PY scripts/summarize.py --root $R --data $D > $R/q_summary.log 2>&1
}

# ---- LPBF Ti-6Al-4V simulation (synthetic/)
[ -f data/synthetic_v1/dataset.h5 ] || $PY -m synthetic.generate --out data/synthetic_v1
[ -f runs/cache/X.npy ]     || $PY scripts/prep_cache.py
[ -f runs/cache/img.npy ]   || $PY scripts/cache_images.py
[ -f runs/teacher/kd_topk.pt ] || $PY scripts/stage1_teacher.py --root runs --data data/synthetic_v1 > runs/q_teacher.log 2>&1
students runs data/synthetic_v1

# ---- circle/crack simulation from the DCCA paper + annotations (dcca_sim/)
[ -f data/dcca_sim/metadata.jsonl ] || $PY -m dcca_sim.generate --out data/dcca_sim --sigma 1.2 2.0 > runs_dcca/q_generate.log 2>&1
[ -f runs_dcca/cache/X.npy ] || $PY dcca_sim/prep.py --sigma 1.2 > runs_dcca/q_prep.log 2>&1
[ -f runs_dcca/teacher/kd_topk.pt ] || $PY scripts/stage1_teacher.py --root runs_dcca --data data/dcca_sim > runs_dcca/q_teacher.log 2>&1
students runs_dcca data/dcca_sim

# ---- annotation-efficiency sweeps
for r in 0.02 0.05 0.25; do
  for R_D in "runs data/synthetic_v1" "runs_dcca data/dcca_sim"; do
    set -- $R_D; R=$1; D=$2; S="$PY scripts/stage3_student.py --root $R --data $D"
    [ -f $R/student/direct_r$r/metrics.json ] || $S --name direct_r$r --decoder base --ratio $r > $R/q_direct_r$r.log 2>&1
    [ -f $R/student/selective_kd_r$r/metrics.json ] || $S --name selective_kd_r$r --align $R/align_selective --kd 1.0 --tok 0.1 --ratio $r > $R/q_selkd_r$r.log 2>&1
  done
done
