import argparse, math, sys, json, time, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mjlab.tasks
import src.tasks.velocity.config.go2_recovery_aligned  # register isolated task
from src.tasks.velocity.config.go2_recovery_aligned.official_go2_limits import (
    POSITION_LOWER, POSITION_UPPER, VELOCITY_LIMIT,
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from dataclasses import asdict

def quat(r,p,y):
    cr, sr = math.cos(r/2), math.sin(r/2); cp, sp=math.cos(p/2),math.sin(p/2); cy,sy=math.cos(y/2),math.sin(y/2)
    return [cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy]

def main():
  ap=argparse.ArgumentParser(); ap.add_argument('--task',default='Unitree-Go2-Recovery-FRNet-Aligned'); ap.add_argument('--checkpoint',required=True); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--horizon',type=int,default=1000); ap.add_argument('--output',default=None); args=ap.parse_args()
  ec=load_env_cfg(args.task, play=True); ec.scene.num_envs=4; ec.viewer=None
  torch.manual_seed(1234)
  # Evaluation must be deterministic: no terrain curriculum/random terrain
  # reset can change the task between runs.
  ec.curriculum = {}
  ec.events.pop('randomize_terrain', None)
  ec.events.pop('push_robot', None)
  # This suite writes its own four fixed fallen poses after reset.
  ec.events.pop('settle_fallen', None)
  if ec.scene.terrain is not None:
      ec.scene.terrain.terrain_type = 'plane'
      ec.scene.terrain.terrain_generator = None
  ec.sim.nconmax = 128
  # Fixed fallen poses create more simultaneous contacts than the default play budget.
  if hasattr(ec.sim, 'nconmax'): ec.sim.nconmax=128
  env=ManagerBasedRlEnv(cfg=ec, device=args.device); wrap=RslRlVecEnvWrapper(env)
  ac=load_rl_cfg(args.task); runner=(load_runner_cls(args.task) or MjlabOnPolicyRunner)(wrap,asdict(ac),device=args.device)
  runner.load(args.checkpoint,load_cfg={'actor':True},strict=True,map_location=args.device); policy=runner.get_inference_policy(device=args.device)
  # A fall direction must rotate the gravity vector in the body frame.  The
  # previous left/right cases used yaw, which leaves the robot upright and
  # therefore was not a recovery test at all.
  robot=env.scene['robot']; poses=[
      ('back',*quat(0,-math.pi/2,0)),
      ('left',*quat(math.pi/2,0,0)),
      ('right',*quat(-math.pi/2,0,0)),
      ('front',*quat(0,math.pi/2,0)),
  ]
  # Root states are world-frame values.  Keep each robot at its own replicated
  # environment origin; placing every instance at (0, 0) makes the fixed-pose
  # suite invalid because the robots overlap.
  root=robot.data.default_root_state.clone()
  root[:,0:3]=env.scene.env_origins.clone()
  root[:,2]+=0.48
  root[:,7:13]=0
  for i,(name,*q) in enumerate(poses): root[i,3:7]=torch.tensor(q,device=args.device)
  env.reset(); robot.write_root_state_to_sim(root); env.sim.forward()
  hold=torch.zeros(4,dtype=torch.int64,device=args.device); first=torch.full((4,),-1,dtype=torch.int64,device=args.device); obs=wrap.get_observations()
  peak_action=torch.zeros(4,device=args.device)
  peak_torque=torch.zeros(4,device=args.device)
  saturation_count=torch.zeros(4,device=args.device)
  sample_count=torch.zeros(4,device=args.device)
  prone_motion_count=torch.zeros(4,device=args.device)
  torque_limits=torch.tensor([23.7,23.7,35.55]*4,device=args.device)
  velocity_limits=torch.tensor(VELOCITY_LIMIT,device=args.device)
  position_lower=torch.tensor(POSITION_LOWER,device=args.device)
  position_upper=torch.tensor(POSITION_UPPER,device=args.device)
  peak_velocity_ratio=torch.zeros(4,device=args.device)
  peak_torque_ratio=torch.zeros(4,device=args.device)
  peak_position_excess=torch.zeros(4,device=args.device)
  velocity_violation_count=torch.zeros(4,device=args.device)
  position_violation_count=torch.zeros(4,device=args.device)
  for t in range(args.horizon):
    with torch.inference_mode(): act=policy(obs); obs,_,done,_=wrap.step(act)
    g=robot.data.projected_gravity_b; z=robot.data.root_link_pos_w[:,2]; ok=(g[:,2] < -0.90)&(z>.30)
    hold=torch.where(ok,hold+1,torch.zeros_like(hold)); first=torch.where((first<0)&(hold>=50),torch.tensor(t,device=args.device),first)
    torque=robot.data.actuator_force.abs()
    peak_action=torch.maximum(peak_action,act.abs().amax(dim=1))
    peak_torque=torch.maximum(peak_torque,torque.amax(dim=1))
    peak_torque_ratio=torch.maximum(peak_torque_ratio,(torque/torque_limits).amax(dim=1))
    joint_velocity=robot.data.joint_vel.abs()
    peak_velocity_ratio=torch.maximum(
        peak_velocity_ratio,(joint_velocity/velocity_limits).amax(dim=1))
    velocity_violation_count+=(joint_velocity>velocity_limits).sum(dim=1)
    position_excess=torch.maximum(
        position_lower-robot.data.joint_pos,
        robot.data.joint_pos-position_upper,
    ).clamp_min(0)
    peak_position_excess=torch.maximum(
        peak_position_excess,position_excess.amax(dim=1))
    position_violation_count+=(position_excess>0.02).sum(dim=1)
    saturation_count+=(torque >= torque_limits*.99).sum(dim=1)
    sample_count+=torque.shape[1]
    prone_motion_count+=((z<.30)&(robot.data.joint_vel.abs().mean(dim=1)>.5)).float()
  results=[]
  for i,(name,*_) in enumerate(poses):
    first_step=int(first[i].item())
    row={'pose':name,'success':bool(first_step>=0),'first_step':first_step,
         'recovery_time_s':None if first_step<0 else round((first_step-49)*env.step_dt,4),
         'hold_steps':int(hold[i].item()),'horizon_steps':args.horizon,
         'peak_action':round(float(peak_action[i]),5),
         'peak_torque_nm':round(float(peak_torque[i]),5),
         'peak_torque_ratio':round(float(peak_torque_ratio[i]),7),
         'peak_velocity_ratio':round(float(peak_velocity_ratio[i]),7),
         'peak_position_excess_rad':round(float(peak_position_excess[i]),7),
         'velocity_limit_violations':int(velocity_violation_count[i]),
         'position_limit_violations_gt_0p02rad':int(position_violation_count[i]),
         'torque_saturation_fraction':round(float(saturation_count[i]/sample_count[i]),7),
         'oscillation_flag':bool(prone_motion_count[i]>=50)}
    row['hardware_safety_pass']=(
        row['peak_torque_ratio']<=1.00001
        and row['peak_velocity_ratio']<=1.0
        and row['peak_position_excess_rad']<=0.02
    )
    results.append(row); print(json.dumps(row,sort_keys=True))
  summary={'task':args.task,'checkpoint':str(args.checkpoint),'seed':1234,'suite':'fixed_four_orientation','results':results,'success_rate':sum(r['success'] for r in results)/len(results),
           'worst_case_success_rate':min(float(r['success']) for r in results),
           'peak_torque_nm':max(r['peak_torque_nm'] for r in results),
           'mean_torque_saturation_fraction':sum(r['torque_saturation_fraction'] for r in results)/len(results),
           'any_oscillation':any(r['oscillation_flag'] for r in results),
           'hardware_safety_pass':all(r['hardware_safety_pass'] for r in results)}
  if args.output:
    Path(args.output).write_text(json.dumps(summary,indent=2)+'\n')
  print(json.dumps({'success_rate':summary['success_rate'],'checkpoint':str(args.checkpoint)},sort_keys=True))
  env.close()
if __name__=='__main__': main()
