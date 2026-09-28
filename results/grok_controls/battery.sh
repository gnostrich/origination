#!/bin/bash
# adversarial algebra controls: same substrate, same extractor, different external tables
cd /home/user/origination
run() { t=$1; steps=$2; python3 -m emergence.grok.run train --arch mlp --seeds 0 --task "$t" --max_steps "$steps" --out results/grok_controls --threads 1 > "results/grok_controls/train_$(echo $t | tr ':.' '-p').log" 2>&1; echo "done $t"; }
export -f run
printf "%s\n" "random:97 12000" "zmod:89 20000" "zmod:101 20000" "zprod:8x8 20000" "sub:97 20000" "scramble:zmod:97 20000" "corrupt:0.05:zmod:97 20000" "corrupt:0.15:zmod:97 20000" | xargs -P 4 -L 1 bash -c 'run $0 $1'
