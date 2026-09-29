#!/bin/bash
cd /home/user/origination
until grep -q "TRAINING DONE" results/world/run_round2.log; do sleep 30; done
run() { python3 -m emergence.grok.run_world train --arch transformer --world $1 --seeds $2 --steps 12000 --ckpt_every 400 --lr 1e-3 --threads 1 --out results/world > results/world/train_transformer_$1_s$2.log 2>&1; echo "done transformer $1 $2"; }
export -f run
printf "%s\n" "perm4 0" "perm4 1" "counter 0" "counter 1" | xargs -P 4 -L 1 bash -c 'run $0 $1'
echo "TRANSFORMER TRAINING DONE"
python3 -m emergence.grok.run_world extract --threads 4 --out results/world > results/world/extract2.log 2>&1
python3 -m emergence.grok.run_world report --out results/world > results/world/report.log 2>&1
echo "ROUND3 DONE"
