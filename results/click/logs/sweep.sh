#!/bin/bash
cd /home/user/origination
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
wd=$1
for seed in 0 1 2; do
  python -m emergence.click.run train --wd $wd --seed $seed && python -m emergence.click.run analyse --wd $wd --seed $seed
done
echo SWEEP_DONE
