"""Execute original reward ASTs without importing Isaac Gym; compare all terms."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import torch
sys.path.insert(0,str(Path(__file__).parent/'go2_recovery_mjlab'))
import baseline_math as m

reference=Path(__file__).parent/'/home/zy/FR-Net/legged_gym/envs/go1'
methods={}
for name in ('legged_robot.py','go2.py','go2_recovery_env.py'):
    tree=ast.parse((/home/zy/FR-Net/legged_gym/envs/go1/name).read_text())
    for cls in [n for n in tree.body if isinstance(n,ast.ClassDef)]:
        for node in cls.body:
            if isinstance(node,ast.FunctionDef) and node.name.startswith('_reward_'):
                namespace={'torch':torch}
                exec(compile(ast.Module(body=[node],type_ignores=[]),name,'exec'),namespace)
                methods[node.name]=namespace[node.name]
torch.manual_seed(55)
n=32
s=NS(num_envs=n,device='cpu',dt=.02,it=750,
     cfg=NS(rewards=NS(base_height_target=.45)))
for key,width in (('base_ang_vel',3),('projected_gravity',3),('root_states',13),
    ('measured_heights',187),('dof_pos',12),('dof_vel',12),('last_dof_vel',12),
    ('actions',12),('last_actions',12),('last_last_actions',12),('torques',12)):
    setattr(s,key,torch.randn(n,width))
s.projected_gravity[:8,2]=-1
s.last_actions[8:16]=0
s.contact_filt=torch.rand(n,4)>.5
s.contact_forces=torch.randn(n,19,3)
s.penalised_contact_indices=[0]
s.feet_indices=[6,10,14,18]
s.dof_pos_limits=torch.tensor([[-1.,1.]]*12)
values={
 'orientation':m.orientation(s.projected_gravity),
 'orientation_pos_gaussian':m.orientation_pos_gaussian(s.projected_gravity),
 'base_height_pos':m.base_height_pos(s.root_states[:,2],s.measured_heights),
 'foot_contact_pos':m.foot_contact_pos(s.contact_filt),
 'step2_target_pos':m.step2_target_pos(s.dof_pos,s.projected_gravity,s.it),
 'ang_vel_xyz':m.squared_norm(s.base_ang_vel),
 'torques':m.squared_norm(s.torques), 'dof_vel':m.squared_norm(s.dof_vel),
 'dof_acc':m.dof_acc(s.last_dof_vel,s.dof_vel,s.dt),
 'dof_pos_limits':m.dof_pos_limits(s.dof_pos,s.dof_pos_limits),
 'collision':m.collision(s.contact_forces[:,s.penalised_contact_indices]),
 'action':m.squared_norm(s.actions),
 'action_rate':m.squared_norm(s.last_actions-s.actions),
 'action_smoothness_2':m.action_smoothness_2(s.actions,s.last_actions,s.last_last_actions),
 'max_velocity':m.max_velocity(s.dof_vel),
 'feet_stumble':m.feet_stumble(s.contact_forces[:,s.feet_indices])}
for name,value in values.items():
    torch.testing.assert_close(value,methods['_reward_'+name](s))
    print('PASS source formula:',name)
tree=ast.parse((/home/zy/FR-Net/legged_gym/envs/go1/'go2_recovery_config.py').read_text())
scales=next(node for node in ast.walk(tree) if isinstance(node,ast.ClassDef) and node.name=='scales')
original={node.targets[0].id:ast.literal_eval(node.value) for node in scales.body if isinstance(node,ast.Assign)}
assert original==m.SCALES
print('PASS all 16 source weights; runtime lifecycle not covered by this test')
