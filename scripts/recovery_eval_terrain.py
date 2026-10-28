"""Four-direction recovery smoke test on representative generated terrains."""
import argparse, json, math, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import mjlab.tasks
import src.tasks.velocity.config.go2_recovery_aligned
from src.tasks.velocity.config.go2_recovery_aligned.official_go2_limits import (
    POSITION_LOWER, POSITION_UPPER, VELOCITY_LIMIT,
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from dataclasses import asdict

def quat(r,p,y=0.0):
    cr,sr=math.cos(r/2),math.sin(r/2); cp,sp=math.cos(p/2),math.sin(p/2); cy,sy=math.cos(y/2),math.sin(y/2)
    return [cr*cp*cy+sr*sp*sy,sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--output',required=True); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--horizon',type=int,default=1000); a=ap.parse_args()
    ec=load_env_cfg('Unitree-Go2-Recovery-FRNet-Aligned',play=True); ec.scene.num_envs=4; ec.viewer=None
    ec.events.pop('settle_fallen',None); ec.events.pop('push_robot',None); ec.events.pop('randomize_terrain',None)
    env=ManagerBasedRlEnv(ec,device=a.device); wrap=RslRlVecEnvWrapper(env)
    ac=load_rl_cfg('Unitree-Go2-Recovery-FRNet-Aligned'); runner=(load_runner_cls('Unitree-Go2-Recovery-FRNet-Aligned') or MjlabOnPolicyRunner)(wrap,asdict(ac),device=a.device); runner.load(a.checkpoint,load_cfg={'actor':True},strict=True,map_location=a.device); policy=runner.get_inference_policy(device=a.device)
    terrain=env.scene.terrain; levels=torch.tensor([2,5,7,8],device=a.device); types=torch.tensor([0,2,5,7],device=a.device)
    terrain.terrain_levels[:]=levels; terrain.terrain_types[:]=types; terrain.env_origins[:]=terrain.terrain_origins[levels,types]
    poses=[('back',quat(0,math.pi/2)),('left',quat(-math.pi/2,0)),('right',quat(math.pi/2,0)),('front',quat(0,-math.pi/2))]
    env.reset(); robot=env.scene['robot']
    # Direct static fallen states: no free-fall, no hidden settling, zero velocity.
    root=robot.data.default_root_state.clone(); root[:,:3]=env.scene.env_origins; root[:,2]+=0.16; root[:,7:]=0
    root[:,3:7]=torch.tensor([q for _,q in poses],device=a.device,dtype=root.dtype)
    robot.write_root_state_to_sim(root); robot.write_root_link_velocity_to_sim(torch.zeros_like(robot.data.root_link_vel_w))
    templates=torch.tensor([[0,1.45,-2.45]*4,[0,1.05,-1.85]*4,[0,1.05,-1.85]*4,[0,1.15,-2.05]*4],device=a.device,dtype=robot.data.joint_pos.dtype)
    q=templates+0.08*torch.randn_like(templates)
    lower=robot.data.joint_pos.new_tensor(POSITION_LOWER)+0.03; upper=robot.data.joint_pos.new_tensor(POSITION_UPPER)-0.03
    robot.write_joint_state_to_sim(q.clamp(lower,upper),torch.zeros_like(robot.data.joint_vel)); env.sim.forward()
    obs=wrap.get_observations()
    hold=torch.zeros(4,dtype=torch.long,device=a.device); first=torch.full((4,),-1,dtype=torch.long,device=a.device); peak=torch.zeros(4,device=a.device); sat=torch.zeros(4,device=a.device); count=torch.zeros(4,device=a.device)
    lim=torch.tensor([23.7]*4+[23.7]*4+[35.55]*4,device=a.device)
    velocity_limits=torch.tensor(VELOCITY_LIMIT,device=a.device)
    position_lower=torch.tensor(POSITION_LOWER,device=a.device)
    position_upper=torch.tensor(POSITION_UPPER,device=a.device)
    peak_torque_ratio=torch.zeros(4,device=a.device); peak_velocity_ratio=torch.zeros(4,device=a.device); peak_position_excess=torch.zeros(4,device=a.device)
    peak_velocity_joint=torch.zeros(4,dtype=torch.long,device=a.device); peak_velocity_step=torch.zeros(4,dtype=torch.long,device=a.device); peak_position_joint=torch.zeros(4,dtype=torch.long,device=a.device); peak_position_step=torch.zeros(4,dtype=torch.long,device=a.device)
    for t in range(a.horizon):
        with torch.inference_mode(): act=policy(obs); obs,_,_,_=wrap.step(act)
        ok=(robot.data.projected_gravity_b[:,2] < -.90)&(robot.data.root_link_pos_w[:,2]>.30); hold=torch.where(ok,hold+1,torch.zeros_like(hold)); first=torch.where((first<0)&(hold>=50),torch.tensor(t,device=a.device),first)
        torque=robot.data.actuator_force.abs(); peak=torch.maximum(peak,torque.amax(1)); sat+=(torque>=lim*.99).sum(1); count+=torque.shape[1]
        peak_torque_ratio=torch.maximum(peak_torque_ratio,(torque/lim).amax(1))
        velocity_ratio_matrix=robot.data.joint_vel.abs()/velocity_limits
        step_velocity,step_velocity_joint=velocity_ratio_matrix.max(1); update_velocity=step_velocity>peak_velocity_ratio
        peak_velocity_joint=torch.where(update_velocity,step_velocity_joint,peak_velocity_joint); peak_velocity_step=torch.where(update_velocity,torch.full_like(peak_velocity_step,t),peak_velocity_step); peak_velocity_ratio=torch.maximum(peak_velocity_ratio,step_velocity)
        position_excess=torch.maximum(position_lower-robot.data.joint_pos,robot.data.joint_pos-position_upper).clamp_min(0)
        step_position,step_position_joint=position_excess.max(1); update_position=step_position>peak_position_excess
        peak_position_joint=torch.where(update_position,step_position_joint,peak_position_joint); peak_position_step=torch.where(update_position,torch.full_like(peak_position_step,t),peak_position_step); peak_position_excess=torch.maximum(peak_position_excess,step_position)
    results=[]
    for i,(pose,_) in enumerate(poses):
        row={'pose':pose,'terrain_column':int(types[i]),'terrain_row':int(levels[i]),'success':bool(first[i]>=0),'recovery_time_s':None if first[i]<0 else round(float((first[i]-49)*env.step_dt),4),'peak_torque_nm':round(float(peak[i]),5),'peak_torque_ratio':round(float(peak_torque_ratio[i]),7),'peak_velocity_ratio':round(float(peak_velocity_ratio[i]),7),'peak_position_excess_rad':round(float(peak_position_excess[i]),7),'peak_velocity_joint':env.scene['robot'].joint_names[int(peak_velocity_joint[i])],'peak_velocity_time_s':round(float(peak_velocity_step[i]*env.step_dt),4),'peak_position_joint':env.scene['robot'].joint_names[int(peak_position_joint[i])],'peak_position_time_s':round(float(peak_position_step[i]*env.step_dt),4),'torque_saturation_fraction':round(float(sat[i]/count[i]),7)}
        row['hardware_safety_pass']=row['peak_torque_ratio']<=1.00001 and row['peak_velocity_ratio']<=1.0 and row['peak_position_excess_rad']<=0.02
        results.append(row)
    slope_max_deg=float(os.environ.get('FRNET_SLOPE_MAX_DEG','5.0')); out={'checkpoint':str(a.checkpoint),'suite':'fixed_four_terrain_types','slope_max_deg':slope_max_deg,'results':results,'success_rate':sum(x['success'] for x in results)/4,'peak_torque_nm':max(x['peak_torque_nm'] for x in results),'hardware_limit_nm':35.55,'hardware_safety_pass':all(x['hardware_safety_pass'] for x in results)}
    Path(a.output).write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2)); env.close()
if __name__=='__main__': main()
