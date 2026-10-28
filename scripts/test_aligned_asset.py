import sys
import mujoco
from mjlab.entity import Entity
try:
    from src.tasks.velocity.config.go2_recovery_aligned.aligned_asset import get_robot_cfg, BODY_NAMES
    from src.tasks.velocity.config.go2_recovery_aligned.official_go2_limits import (
        FRNET_EFFORT_LIMIT, OFFICIAL_EFFORT_LIMIT, JOINT_NAMES,
        POSITION_LOWER, POSITION_UPPER, VELOCITY_LIMIT)
except ImportError:
    from frnet_work.aligned_asset import get_robot_cfg, BODY_NAMES
    from frnet_work.official_go2_limits import (
        FRNET_EFFORT_LIMIT, OFFICIAL_EFFORT_LIMIT, JOINT_NAMES,
        POSITION_LOWER, POSITION_UPPER, VELOCITY_LIMIT)
robot=Entity(get_robot_cfg(sys.argv[1]))
model=robot.spec.compile()
names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i) for i in range(model.nbody)]
assert all(name in names for name in BODY_NAMES)
assert model.nu == 12
assert model.nq == 19 and model.nv == 18
for name, lower, upper in zip(JOINT_NAMES, POSITION_LOWER, POSITION_UPPER):
    joint_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
    assert joint_id >= 0
    assert abs(model.jnt_range[joint_id,0]-lower)<1e-6
    assert abs(model.jnt_range[joint_id,1]-upper)<1e-6
effort_by_joint={}
for actuator_id in range(model.nu):
    joint_id=int(model.actuator_trnid[actuator_id,0])
    joint_name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,joint_id)
    effort_by_joint[joint_name]=float(model.actuator_forcerange[actuator_id,1])
for name, effort, official in zip(JOINT_NAMES,FRNET_EFFORT_LIMIT,OFFICIAL_EFFORT_LIMIT):
    assert abs(effort_by_joint[name]-effort)<1e-6
    assert effort<=official
assert VELOCITY_LIMIT == (30.1,30.1,15.70)*4
# go2_head.xml carries the physical head load in the base/head bodies.
assert abs(model.body_mass[names.index("base")]-7.505)<1e-6
assert abs(model.body_mass[names.index("Head_lower")]-0.001)<1e-9
assert abs(model.body_mass[names.index("xiaoli_head")]-1.6)<1e-6
assert abs(float(model.body_mass.sum())-17.391408)<1e-6
for leg in ("FL","FR","RL","RR"):
    # Foot bodies are intentionally massless; source collision spheres are on calves.
    assert abs(model.body_mass[names.index(f"{leg}_foot")])<1e-9
assert len(set(BODY_NAMES)) == 19
print('PASS: Go2 URDF body/masses, official position limits, source torque saturation within official envelope, free base, 12 explicit PD actuators')
