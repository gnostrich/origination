#!/bin/bash
cd /home/user/origination
until [ "$(grep -c '^done' results/world_tf/run.log)" -ge 2 ]; do sleep 20; done
# stop the slow queued extraction and the queue script itself
python3 - <<'PY'
import os, signal
for pid in os.listdir('/proc'):
    if not pid.isdigit(): continue
    try: argv = open(f'/proc/{pid}/cmdline','rb').read().split(b'\0')
    except Exception: continue
    if (len(argv) > 3 and argv[1] == b'-m' and argv[2] == b'emergence.grok.run_world' and argv[3] == b'extract' and b'results/world_tf' in argv) \
       or (len(argv) > 1 and argv[-1].endswith(b'results/world_tf/run.sh')):
        os.kill(int(pid), signal.SIGTERM); print('killed', pid, argv[:4])
PY
python3 -m emergence.grok.run_world extract --threads 4 --max_ckpts 6 --force --out results/world_tf > results/world_tf/extract.log 2>&1
python3 -m emergence.grok.run_world report --out results/world_tf > results/world_tf/report.log 2>&1
echo "FAST TAIL DONE"
