#!/bin/bash
cd /home/user/origination
run() { python3 -m emergence.grok.run_world train --world $1 --seeds $2 --steps 6000 --ckpt_every 200 --threads 1 --out results/world > results/world/train_$1_s$2.log 2>&1; echo "done $1 $2"; }
export -f run
printf "%s\n" "perm4 0" "perm4 1" "perm4 2" "perm4reset 0" "perm4reset 1" | xargs -P 4 -L 1 bash -c 'run $0 $1'
python3 -m emergence.grok.run_world extract --threads 4 --out results/world > results/world/extract.log 2>&1
python3 -m emergence.grok.run_world report --out results/world > results/world/report.log 2>&1
echo "ALL DONE"
