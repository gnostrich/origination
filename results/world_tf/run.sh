#!/bin/bash
cd /home/user/origination
run() { python3 -m emergence.grok.run_world train --arch transformer --world perm4 --seeds $1 --n_train 4096 --n_test 1024 --steps 8000 --ckpt_every 400 --lr 1e-3 --threads 2 --out results/world_tf > results/world_tf/train_perm4_s$1.log 2>&1; echo "done $1"; }
export -f run
printf "%s\n" 0 1 | xargs -P 2 -L 1 bash -c 'run $0'
python3 -m emergence.grok.run_world extract --threads 4 --out results/world_tf > results/world_tf/extract.log 2>&1
python3 -m emergence.grok.run_world report --out results/world_tf > results/world_tf/report.log 2>&1
echo "TF DONE"
