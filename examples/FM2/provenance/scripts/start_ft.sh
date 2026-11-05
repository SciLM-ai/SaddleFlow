#!/bin/bash
# The full-data fine-tune of FM1 with the recipe the user picks (18, 19 or 20): FM1's data mix (oc20 x2, oc22 x8, mp20bat x8,
# lemat x1; oc22 -> oc20 task), FM1's released weights, 1 epoch, 64 x 8 = batch 512 as FM1, Huber, backbone LR 3e-3, line part
# (u0 0.5), mover-set mix (20 % all atoms, else threshold U(0, 1) A), CoM loss + --align-com; plus the recipe's one change:
#   18: none   19: ball sets (20 % ball r U(3, 5) A around the most-moving atom)   20: x_t perturbation 0.1 A
# usage (on the job's first node, once the nodes are free): nohup setsid bash start_ft.sh 18|19|20 > log 2>&1 &
source /scratch/00405/graeme/saddleflow/campaigns/2026-10-17_al_scratch/oc22_64b/night_lib.sh
cd /scratch; R=${1:?18, 19 or 20}
case $R in 18) X="";; 19) X="--mover-ball-prob 0.2 --mover-ball-rmin 3 --mover-ball-rmax 5";; 20) X="--xt-perturb-sigma 0.1";; *) exit 1;; esac
FT="--backbone uma-m-1p1 --uma-lr 3e-3 --early-time-film-blocks 0,1,2,3,4,5,6,7,8,9 --loss-type huber --huber-delta 0.05 --mover-all-prob 0.2 --mover-thresh-max 1.0 --align-com --init-weights $REL $X"
O=$A/full_64b/ft_run${R}m
log "FULL fine-tune ft_run${R}m (FM1 weights + run${R}m's recipe on FM1's data mix, 1 epoch, 64 x 8, CoM loss + align-com) on nodes 0..63 -> full_64b/ft_run${R}m"
LINE=0.5 BS=8 EPOCHS=1 PORT=29511 EXTRA="$FT" bash $B/launch_full_slot_d.sh gen 0 64 $O > $O.launch.log 2>&1
log "FULL fine-tune ft_run${R}m stopped ($(grep -o 'done. total time.*' $O/gen.log || tail -n 1 $O/gen.log | cut -c1-150))"
