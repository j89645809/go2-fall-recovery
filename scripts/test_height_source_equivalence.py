import ast
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import torch
sys.path.insert(0,str(Path(__file__).parent/'go2_recovery_mjlab'))
from aligned_heights import local_grid,sample_heights

def reference_yaw(q,v):
    q=q.clone().reshape(-1,4);q[:,:2]=0
    q=q/q.norm(dim=-1,keepdim=True).clamp_min(1e-9)
    v=v.reshape(-1,3)
    xyz=q[:,:3]
    t=2*torch.cross(xyz,v,dim=-1)
    return (v+q[:,3:4]*t+torch.cross(xyz,t,dim=-1)).reshape(-1,187,3)

tree=ast.parse((Path(__file__).parent/'frnet_/home/zy/FR-Net/legged_gym/envs/base/legged_robot.py').read_text())
node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_get_heights')
scope={'torch':torch,'quat_apply_yaw':reference_yaw}
exec(compile(ast.Module(body=[node],type_ignores=[]),'original_heights','exec'),scope)
torch.manual_seed(55)
for border in (0.,5.):
    n=32
    pos=torch.randn(n,3)*4
    q=torch.randn(n,4);q=q/q.norm(dim=-1,keepdim=True)
    field=torch.randint(-100,100,(103,109),dtype=torch.int16)
    state=NS(cfg=NS(terrain=NS(mesh_type='trimesh',horizontal_scale=.1,vertical_scale=.005)),
        terrain=NS(cfg=NS(border_size=border,horizontal_scale=.1,vertical_scale=.005)),height_samples=field,
        base_quat=q[:,[1,2,3,0]],height_points=local_grid(pos).repeat(n,1,1),
        root_states=pos,num_envs=n,num_height_points=187)
    actual=sample_heights(pos,q,field,.1,.005,border)
    expected=scope['_get_heights'](state)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
print('PASS: source height sampling exact for random rotations, edge clipping and 187-point ordering')
