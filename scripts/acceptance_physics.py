"""Validated per-case physical parameters for vectorized acceptance."""
import torch

def set_case_friction(env, robot_geom_ids, friction):
    if friction.shape != (env.num_envs,) or not torch.isfinite(friction).all() or not (friction > 0).all():
        raise ValueError("Expected one finite positive sliding coefficient per environment")
    # Use the simulation API: it updates Warp storage, bridge and CUDA graphs.
    # A tensor clone alone would leave the physics engine using the old array.
    if "geom_friction" not in env.sim.expanded_fields:
        env.sim.expand_model_fields(("geom_friction",))
    field = env.sim.model.geom_friction
    if env.num_envs > 1 and (field.stride(0) == 0 or field.wp_array.shape[0] != env.num_envs):
        raise RuntimeError("Friction storage is still shared across environments")
    field[:, robot_geom_ids, 0] = friction[:, None]
    if friction.is_cuda:
        torch.cuda.synchronize(friction.device)
    actual = env.sim.model.geom_friction[:, robot_geom_ids, 0]
    expected = friction[:, None].expand_as(actual)
    if not torch.equal(actual, expected):
        raise RuntimeError("Simulator friction does not match the requested per-case coefficients")
    return {"per_environment_storage": True, "requested_values": friction.unique().tolist(),
            "max_assignment_error": float((actual-expected).abs().max())}
