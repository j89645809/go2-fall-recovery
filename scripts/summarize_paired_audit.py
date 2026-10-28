import json, torch
from pathlib import Path
root=Path('run_logs/paired_audit')
basepath=Path('logs/rsl_rl/go2_recovery_frnet_aligned')
b=torch.load(basepath/'2026-10-11_16-03-28_acceptance_v32_backend_static_reset_20/model_14475.pt',map_location='cpu',weights_only=False)['actor_state_dict']
z=torch.load(basepath/'residual_baseline_init/model_14475.pt',map_location='cpu',weights_only=False)['actor_state_dict']
assert all(torch.equal(v,z['teacher_'+k]) for k,v in b.items() if k.startswith('mlp.'))
assert torch.count_nonzero(z['mlp.6.weight'])==0 and torch.count_nonzero(z['mlp.6.bias'])==0
summary=[]
for seed in (1234,2345,3456):
    a=json.loads((root/f'baseline_{seed}.json').read_text())
    c=json.loads((root/f'zero_{seed}.json').read_text())
    assert a['summary']['protocol']==c['summary']['protocol']=='strict_static_repeated_v3'
    assert a['summary']['static_placement']['invalid_cases']==c['summary']['static_placement']['invalid_cases']==0
    gap=c['summary']['success_rate']-a['summary']['success_rate']
    summary.append(dict(seed=seed,baseline=a['summary']['success_rate'],zero=c['summary']['success_rate'],gap=gap))
gate=all(abs(s['gap'])<=0.03 for s in summary)
out=dict(action_equivalence_pass=True,rollout_screen_pass=gate,comparisons=summary,note='Rollouts are not bitwise reproducible. 3pp is a screening threshold, not statistical equivalence proof. Seed1234 supplies retention data; seeds2345/3456 are validation.')
(root/'summary.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
if not gate: raise SystemExit('Baseline/zero repeatability screen failed; training not started')
