#!/bin/bash
# Wait for the 64-node job, then launch the foundation run ON THAT JOB's head node over ssh, detached (setsid/nohup),
# so the run belongs to the 64-node job and survives the loss of the session that started this watcher.
# After launch, record (timestamp, latest logged step) every 2 min for step-time / ETA estimates.
#   usage: watch_and_launch.sh <JOBID> <RUN_NAME>
set -u
J=$1; NAME=$2
C=/scratch/08405/ilgar/saddleflow/campaigns/2026-09-29_v2_foundation
OUT=$C/runs/$NAME; LOG=$C/logs/$NAME.log; W=$C/logs/watch_$NAME.log; PROG=$C/logs/${NAME}_progress.tsv
echo "$(date) watching job $J" >> $W
while true; do
  st=$(squeue -h -j $J -o %T 2>/dev/null)
  if [ "$st" = "RUNNING" ]; then break; fi
  if [ -z "$st" ]; then echo "$(date) job $J no longer in the queue; not launching" >> $W; exit 1; fi
  sleep 30
done
NL=$(squeue -h -j $J -o %N); HEAD=$(scontrol show hostnames "$NL" | head -n1); NN=$(scontrol show hostnames "$NL" | wc -l)
echo "$(date) job $J RUNNING on $NN nodes, head $HEAD: $NL" >> $W
sleep 60                                            # let the allocation settle
if [ -e "$OUT/config.json" ]; then echo "$(date) $OUT already exists; not launching twice" >> $W; exit 1; fi
ssh -o BatchMode=yes $HEAD "nohup setsid bash $C/scripts/launch_foundation.sh $J '$NL' $OUT > $LOG 2>&1 < /dev/null & disown; exit"
echo "$(date) launched on $HEAD -> $OUT (log $LOG)" >> $W
printf "time\tstep\n" > $PROG
while squeue -h -j $J -o %T 2>/dev/null | grep -q RUNNING; do
  s=$(grep -o "step *[0-9]* mode" $LOG 2>/dev/null | tail -1 | grep -o "[0-9]*")
  printf "%s\t%s\n" "$(date +%s)" "${s:-NA}" >> $PROG
  grep -q "done. total time" $LOG 2>/dev/null && { echo "$(date) training finished" >> $W; break; }
  sleep 120
done
echo "$(date) watcher exiting" >> $W
