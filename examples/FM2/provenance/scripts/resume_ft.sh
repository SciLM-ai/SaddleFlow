#!/bin/bash
# Resume the full-data fine-tune ft_run<R>m (start_ft.sh) in a new 64-node job (same 64 x 8 layout); run on the job's first node:
#   nohup setsid bash resume_ft.sh <JOBID> 19 > <log> 2>&1 &
# Latest complete checkpoint_step (a half-written one at the old job's end is skipped); earlier gen.log kept as gen_partN.log;
# same settings as start_ft.sh except --init-weights (the checkpoint already holds the weights, optimizer, schedule and EMA).
export JOBID=${1:?usage: resume_ft.sh JOBID 18|19|20}; R=${2:?recipe}
source /scratch/00405/graeme/saddleflow/campaigns/2026-10-17_al_scratch/oc22_64b/night_lib.sh
cd /scratch; O=$A/full_64b/ft_run${R}m
case $R in 18) X="";; 19) X="--mover-ball-prob 0.2 --mover-ball-rmin 3 --mover-ball-rmax 5";; 20) X="--xt-perturb-sigma 0.1";; *) exit 1;; esac
FT="--backbone uma-m-1p1 --uma-lr 3e-3 --early-time-film-blocks 0,1,2,3,4,5,6,7,8,9 --loss-type huber --huber-delta 0.05 --mover-all-prob 0.2 --mover-thresh-max 1.0 --align-com $X"
CK=""
for d in $(ls -d $O/gen/checkpoint_step_* | sort -r); do
  if [ -f $d/meta.json ] && [ -f $d/ema.pt ] && [ -f $d/model.safetensors ] && [ -f $d/optimizer.bin ] && [ -f $d/scheduler.bin ] \
     && python -c "import json, torch; json.load(open('$d/meta.json')); torch.load('$d/ema.pt', map_location='cpu')" 2>/dev/null; then CK=$d; break; fi
done
[ -n "$CK" ] || { log "ERROR resume_ft: no complete checkpoint under $O/gen"; exit 1; }
k=1; while [ -f $O/gen_part$k.log ]; do k=$((k + 1)); done; cp $O/gen.log $O/gen_part$k.log
log "FULL fine-tune ft_run${R}m resumed from $(basename $CK) ($(cat $CK/meta.json)) in job $JOBID on nodes 0..63"
LINE=0.5 BS=8 EPOCHS=1 PORT=29512 EXTRA="$FT --resume-from $CK" bash $B/launch_full_slot_d.sh gen 0 64 $O > $O.resume$k.launch.log 2>&1
log "FULL fine-tune ft_run${R}m stopped ($(grep -o 'done. total time.*' $O/gen.log || tail -n 1 $O/gen.log | cut -c1-150))"
