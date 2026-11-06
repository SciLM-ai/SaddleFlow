#!/bin/bash
# On the CPU session node: when the next 64-node job (default 1099377) runs, start resume_ft.sh <job> 19 on its first node
# (detached there, so it lives as long as that GPU job).
A=/scratch/00405/graeme/saddleflow/campaigns/2026-10-17_al_scratch; B=$A/oc22_64b; JOB=${1:-1099377}
log() { echo "$(date '+%m-%d %H:%M') $*" >> $A/progress.log; }
until [ "$(squeue -j $JOB -h -o %T 2>/dev/null)" = RUNNING ]; do
  [ -z "$(squeue -j $JOB -h -o %T 2>/dev/null)" ] && { log "ERROR start_resume_ft: job $JOB is gone (not running, not pending)"; exit 1; }
  sleep 60
done
N0=$(scontrol show hostnames "$(squeue -j $JOB -h -o %N)" | head -1)
log "64-node job $JOB RUNNING (first node $N0, $(squeue -j $JOB -h -o %D) nodes); resuming ft_run19m there"
sleep 60
ssh -f -o BatchMode=yes $N0 "cd /scratch && nohup setsid bash $B/resume_ft.sh $JOB 19 > $A/logs/resume_ft_$JOB.log 2>&1 < /dev/null &" \
  || log "ERROR start_resume_ft: could not ssh to $N0"
