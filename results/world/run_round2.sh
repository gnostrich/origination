#!/bin/bash
cd /home/user/origination
run() { python3 -m emergence.grok.run_world train --world $1 --seeds $2 --steps $3 --ckpt_every 200 --threads 1 --out results/world > results/world/train_$1_s$2.log 2>&1; echo "done $1 $2"; }
export -f run
printf "%s\n" "counter 0 6000" "counter 1 6000" "counter 2 6000" "perm4reset 0 12000" "perm4reset 1 12000" | xargs -P 4 -L 1 bash -c 'run $0 $1 $2'
echo "TRAINING DONE"
