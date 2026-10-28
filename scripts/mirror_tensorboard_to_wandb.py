#!/usr/bin/env python3
"""Mirror an active TensorBoard run to one resumable Weights & Biases run."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import wandb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--log-dir',type=Path,required=True)
    parser.add_argument('--entity',required=True)
    parser.add_argument('--project',required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--name',required=True)
    parser.add_argument('--watch-pid',type=int)
    parser.add_argument('--interval',type=int,default=30)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    log_dir=args.log_dir.resolve()
    if not any(log_dir.glob('events.out.tfevents.*')):
        raise FileNotFoundError(f'No TensorBoard events in {log_dir}')
    state_path=log_dir/'.wandb_mirror_state.json'
    last_step=json.loads(state_path.read_text()).get('last_step',-1) if state_path.exists() else -1
    run=wandb.init(entity=args.entity,project=args.project,id=args.run_id,
                   name=args.name,dir=str(log_dir),resume='allow',
                   tags=('FR-Net','Go2','success-replay','teacher-anchored'),
                   config={'source_checkpoint':'model_13798.pt',
                           'tensorboard_log_dir':str(log_dir),
                           'success_replay_fraction':0.60,
                           'targeted_fraction':0.40,
                           'algorithm':'teacher-anchored PPO',
                           'learning_rate':1e-4})
    print('WANDB_RUN_URL',run.url,flush=True)
    idle_after_exit=0
    try:
        while True:
            accumulator=EventAccumulator(str(log_dir),size_guidance={'scalars':0})
            accumulator.Reload()
            by_step={}
            for tag in accumulator.Tags().get('scalars',[]):
                # These two TensorBoard convenience plots use wall-time
                # counters, while all training metrics use PPO iteration.
                if tag.endswith('/time'):
                    continue
                for event in accumulator.Scalars(tag):
                    if event.step>last_step:
                        by_step.setdefault(event.step,{})[tag]=event.value
            for step,metrics in sorted(by_step.items()):
                run.log(metrics,step=step)
                last_step=step
                temp=state_path.with_suffix('.tmp')
                temp.write_text(json.dumps({'run_id':args.run_id,'last_step':last_step,'url':run.url}))
                temp.replace(state_path)
                print('SYNCED_STEP',step,'METRICS',len(metrics),flush=True)
            if args.once: break
            if args.watch_pid and not Path(f'/proc/{args.watch_pid}').exists():
                idle_after_exit=idle_after_exit+1 if not by_step else 0
                if idle_after_exit>=2: break
            time.sleep(args.interval)
    finally:
        run.finish()


if __name__=='__main__': main()
