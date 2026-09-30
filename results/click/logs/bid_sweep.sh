#!/bin/bash
# behavioural interface discovery over existing checkpoints, one weight-decay level per process
cd /home/user/origination
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
wd=$1
for seed in 0 1 2; do python -m emergence.click.run_bid search --wd $wd --seed $seed; done
for seed in 0 1 2; do python -m emergence.click.run_bid search --wd $wd --seed $seed --null; done
echo BID_DONE
