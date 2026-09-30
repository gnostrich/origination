#!/bin/bash
cd /home/user/origination
run() { python3 -m emergence.grok.resolution --system $1 --run $2 --n_ckpt 6 --threads 1 --out results/resolution > results/resolution/$(basename $2).log 2>&1; echo "done $2"; }
export -f run
printf "%s\n" "world results/world/perm4_s0" "world results/world/perm4_s1" "world results/world/counter_s0" "world results/world/counter_s1" \
  "grok results/grok/mlp_s0" "grok results/grok/transformer_s0" \
  "discover results/discover/g2.5_t0/student_s0" "discover results/discover/g2.5_t0/student_s1" "discover results/discover/g1.0_t0/student_s0" \
  | xargs -P 4 -L 1 bash -c 'run $0 $1'
echo "RES DONE"
