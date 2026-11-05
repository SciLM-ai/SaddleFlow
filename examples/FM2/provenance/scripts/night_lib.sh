# Helpers for 64-node job JOBID (default 1095531), sourced by chain_b.sh; run on a node of the job. Code: code_snap_1024.
A=/scratch/00405/graeme/saddleflow/campaigns/2026-10-17_al_scratch; B=$A/oc22_64b; S=$A/scripts; F=$B/farm1.sh
export JOBID=${JOBID:-1095531}
REF=$A/oc22_64/run1/sella/x1K20
REL=/scratch/00405/graeme/saddleflow/campaigns/2026-10-15_qsweep/caches/hf/hub/models--SciLM--SaddleFlow/snapshots/f003c2699ac1b9eb0446fa128f9f34c249af9757
SEL=$A/oc22_64/run1/sel/checkpoint_final
source $B/msenv.sh
log() { echo "$(date '+%m-%d %H:%M') $*" >> $A/progress.log; }

# sella_check <first node> <n nodes> <run dir under oc22_64b> <tag> "<gen args>": ms_sella_gen.py on run1's 402 records,
# 1 draw each; endpoints on the first node, Sella split over the n nodes (records i % n).
sella_check() {
  local O=$B/$3/sella/$4 i; mkdir -p $O
  log "oc22 Sella check of $3 ($4, run1's 402 records) on nodes $1..$(( $1 + $2 - 1 ))"
  bash $F $1 $O.gen.log "python $S/ms_sella_gen.py --phase gen --ckpt $B/$3/gen/checkpoint_final $5 --out $O --max-records 1000"
  for (( i = 0; i < $2; i++ )); do
    bash $F $(( $1 + i )) $O.sella$i.log "python -u $S/ms_sella_gen.py --phase sella --out $O --workers 64 --only-done $REF --shard $i --nshards $2" &
  done
  wait
  log "[oc22 Sella check, $3 $4] $(cd /scratch && python $S/ms_sella_gen.py --summary --out $O 2>&1 | tr '\n' ' ' | cut -c1-1200)"
}

# discover <name> <first node> <n nodes> "<gen args>": ms_discover.py, 100 test minima x 10 draws, Sella split over n nodes.
discover() {
  local O=$B/discover/$1 i; mkdir -p $O
  log "oc22 discovery head-to-head arm $1 on nodes $2..$(( $2 + $3 - 1 ))"
  bash $F $2 $O.gen.log "python $S/ms_discover.py --phase gen $4 --out $O"
  for (( i = 0; i < $3; i++ )); do
    bash $F $(( $2 + i )) $O.sella$i.log "python -u $S/ms_discover.py --phase sella --out $O --shard $i --nshards $3 --workers 64" &
  done
  wait
  log "[oc22 discover, $1] $(cd /scratch && python $S/ms_discover.py --summary --out $O 2>&1 | tr '\n' ' ' | cut -c1-900)"
}

# al_run <first node> <n nodes> <name> <starts npz> <ckpt> <known|sym> <label>: al_known.py 5 x 4 (task oc20), split over n
# nodes, merged; scored as Al test 1 (known_set_score.py) or as archive saddles found (score_sym.py, first 2,000).
al_run() {
  local O=$A/eval_b/$3 i; mkdir -p $O
  log "Al $3 on nodes $1..$(( $1 + $2 - 1 ))"
  for (( i = 0; i < $2; i++ )); do
    bash $F $(( $1 + i )) $O.s$i.log "python -u $S/al_known.py --starts $4 --ckpt $5 --passes 5 --steps 4 --shard $i --nshards $2 --out $O" &
  done
  wait
  python $S/al_known.py --starts $4 --out $O --merge
  if [ "$6" = known ]; then log "[$7] $(cd $S && python known_set_score.py $O $4 2>&1 | tr '\n' ' ' | cut -c1-700)"
  else log "[$7] $(cd $S && python score_sym.py $O --n 2000 2>&1 | tr '\n' ' ' | cut -c1-600)"; fi
}
