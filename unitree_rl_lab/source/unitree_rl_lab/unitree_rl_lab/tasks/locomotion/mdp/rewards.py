from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Tuple

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


from isaaclab.envs import ManagerBasedEnv
from isaaclab.sensors import RayCaster
"""
Joint penalties.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


# --- Crouch reference pose (radians) ---
CROUCH_DICT = {
    "FL_hip_joint":  0.0392,
    "FL_thigh_joint": 1.15,
    "FL_calf_joint": -2.33,

    "FR_hip_joint": -0.0392,
    "FR_thigh_joint": 1.15,
    "FR_calf_joint": -2.33,

    "RL_hip_joint":  0.0713,
    "RL_thigh_joint": 1.15,
    "RL_calf_joint": -2.33,

    "RR_hip_joint": -0.0713,
    "RR_thigh_joint": 1.15,
    "RR_calf_joint": -2.33,
}
# （可選）快取，避免每 step 都重建 tensor
_CROUCH_TENSOR_CACHE = {}   # key: tuple(joint_names) -> torch.Tensor (CPU)


def joint_position_penalty_crouch(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)

    names = tuple(asset.data.joint_names)  # 讓它可 hash
    q_ref_cpu = _CROUCH_TENSOR_CACHE.get(names, None)
    if q_ref_cpu is None:
        missing = [n for n in names if n not in CROUCH_DICT]
        if missing:
            raise RuntimeError(f"CROUCH_DICT missing joints: {missing}")
        # 先建在 CPU，之後每次搬到 GPU（或你也可以 cache per device）
        q_ref_cpu = torch.tensor([CROUCH_DICT[n] for n in names], dtype=asset.data.joint_pos.dtype, device="cpu")
        _CROUCH_TENSOR_CACHE[names] = q_ref_cpu

    q_ref = q_ref_cpu.to(device=asset.data.joint_pos.device)  # (J,)
    reward = torch.linalg.norm(asset.data.joint_pos - q_ref[None, :], dim=1)

    return torch.where(
        torch.logical_or(cmd > 0.2, body_vel > velocity_threshold),
        reward,
        stand_still_scale * reward,
    )

# default
def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


"""
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


"""
Other rewards.
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward
    
# ---- reuse quaternion helpers (copy from observations.py or import if you prefer) ----
def _quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)

def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack([w, x, y, z], dim=-1)

def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qv = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
    return _quat_mul(_quat_mul(q, qv), _quat_conj(q))[..., 1:4]

def _quat_rotate_inv(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return _quat_rotate(_quat_conj(q), v)


def _get_contact_forces(contact: ContactSensor) -> torch.Tensor:
    """
    Return contact forces tensor shaped (N, B, 3) if possible.
    Try multiple field names to be robust across IsaacLab versions.
    """
    d = contact.data
    for name in ("net_forces_w", "contact_forces_w", "forces_w"):
        t = getattr(d, name, None)
        if t is not None:
            return t
    raise AttributeError("Cannot find contact forces in ContactSensor.data (tried net_forces_w/contact_forces_w/forces_w).")


def _get_contact_forces_history(contact: ContactSensor):
    """
    Optional: return history forces shaped (N, T, B, 3) if available.
    """
    d = contact.data
    for name in ("net_forces_w_history", "contact_forces_w_history", "forces_w_history"):
        t = getattr(d, name, None)
        if t is not None:
            return t
    return None

def _dbg_grid_stats(env, tag: str, grid: torch.Tensor, fill_value: float, every: int = 200) -> None:
    """Print grid stats for env0 every `every` steps."""
    if not hasattr(env, "_dbg_counter_rew"):
        env._dbg_counter_rew = 0
    env._dbg_counter_rew += 1
    if env._dbg_counter_rew % every != 0:
        return

    g0 = grid[0]  # env0
    fill_ratio = (g0 == fill_value).float().mean().item()
    print(
        f"[DBG][{tag}] grid0 mean/std/min/max = "
        f"{g0.mean().item():.6f} {g0.std().item():.6f} {g0.min().item():.6f} {g0.max().item():.6f} | "
        f"fill_ratio={fill_ratio:.3f}"
    )

def _build_front_height_grid(env, sensor_cfg, offset, x_range, y_range, grid_shape, fill_value):
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    # 跟 obs 完全一致：平面≈0，坑洞更負
    heights = sensor.data.ray_hits_w[..., 2] - sensor.data.pos_w[:, 2].unsqueeze(1) + offset  # (N,R)

    H, W = grid_shape
    if heights.shape[1] != H * W:
        raise RuntimeError(f"[front_grid] Expect {H*W} rays, got {heights.shape[1]}.")

    #_dbg_grid_stats(env, "REW/front_grid", heights, fill_value, every=1)
    return heights.view(heights.shape[0], H, W)


def _local_std_map(grid: torch.Tensor, k: int = 3) -> torch.Tensor:
    """
    grid: (N,H,W)
    return: (N,H,W) local std (approx) using unfold.
    """
    N, H, W = grid.shape
    pad = k // 2
    x = torch.nn.functional.pad(grid.unsqueeze(1), (pad, pad, pad, pad), mode="replicate")  # (N,1,H+2p,W+2p)
    patches = x.unfold(2, k, 1).unfold(3, k, 1)  # (N,1,H,W,k,k)
    patches = patches.contiguous().view(N, 1, H, W, k * k)
    mean = patches.mean(dim=-1)
    var = (patches - mean.unsqueeze(-1)).pow(2).mean(dim=-1)
    std = torch.sqrt(var + 1e-8).squeeze(1)
    return std


def foothold_touchdown_reward(
    env: ManagerBasedEnv,
    height_sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    offset: float = 0.5,
    x_range: Tuple[float, float] = (0.0, 0.8),
    y_range: Tuple[float, float] = (-0.4, 0.4),
    grid_shape: Tuple[int, int] = (9, 9),
    fill_value: float = -10.0,
    gap_height_th: float = -0.5,
    flat_std_th: float = 0.08,
    contact_force_th: float = 5.0,
    command_name: str = "base_velocity",
    min_cmd_x: float = 0.3,
    min_base_vx: float = 0.1,
    # --- gap activation (修法A) ---
    gap_present_th: float = 0.10,     # ratio gate（保留）
    gap_near_th: float = 0.25,        # ✅ 只有 gap 出現在前方 <= 25cm 才啟動
    front_y_th: float = 0.15,         # ✅ 只看正前方 y ∈ [-0.12, 0.12]
    # shaping (rate; 最後外面會 *dt)
    progress_w: float = 2.0,         # reward rate per (m/s)
    stuck_w: float = 2.0,             # penalty rate
    back_w: float = 10.0,             # penalty rate per (m/s) backward
    # milestone jackpot (event)
    cross_dist: float = 0.25,         # 連續 stones：milestone 間距建議 0.20~0.40
    cross_bonus: float = 10.0,        # ✅ 一次觸發總共加多少（函式內會 /dt）
    cross_min_vx: float = 0.15,
    cross_good_ratio_th: float = 0.60,
    cross_x_th: float = 0.25,     # 腳要落在 base frame 前方 >= 25cm
    cross_y_th: float = 0.12,     # 腳要落在中線 |y| < 12cm（逼直走）
    yaw_dev_th: float = 0.25,     # 與看到 gap 當下的 yaw 偏差 < 0.25 rad
    yaw_rate_th: float = 1.0,     # yaw rate < 1.0 rad/s
    cmd_y_th: float = 0.10,       # 側向命令要小（防鑽漏洞）
) -> torch.Tensor:

    dt = float(getattr(env, "step_dt", 0.02))

    # -----------------------------
    # 1) height grid & masks
    # -----------------------------
    grid = _build_front_height_grid(env, height_sensor_cfg, offset, x_range, y_range, grid_shape, fill_value)  # (N,H,W)
    std_map = _local_std_map(grid, k=3)

    is_gap = grid < gap_height_th
    is_flat = std_map < flat_std_th
    is_good = (~is_gap) & is_flat

    N, H, W = grid.shape
    device = grid.device

    gap_ratio = is_gap.float().mean(dim=(1, 2))          # (N,)
    good_ratio = is_good.float().mean(dim=(1, 2))        # (N,)

    # -----------------------------
    # 1.5) 修法A：gap 必須「正前方 + 近距離」才算 present
    # -----------------------------
    x0, x1 = x_range
    y0, y1 = y_range
    dx_cell = (x1 - x0) / (W - 1)

    # only consider rows near centerline
    y_vals = torch.linspace(y0, y1, H, device=device)          # (H,)
    front_rows = (y_vals.abs() <= front_y_th)                  # (H,)

    # any gap in each x-column within front_rows
    gap_any_x = is_gap[:, front_rows, :].any(dim=1)            # (N,W)

    idxs = torch.arange(W, device=device).view(1, W).expand(N, W)
    big = torch.full_like(idxs, W)
    min_ix = torch.where(gap_any_x, idxs, big).min(dim=1).values   # (N,)  (W means no gap)

    x_gap_near = x0 + min_ix.float() * dx_cell                    # (N,)
    gap_close = (min_ix < W) & (x_gap_near <= gap_near_th)         # (N,)

    gap_present = (gap_ratio >= gap_present_th) & gap_close        # (N,)

    # -----------------------------
    # 2) robot state (Tensor)
    # -----------------------------
    robot = env.scene[asset_cfg.name]
    base_pos_w = robot.data.root_pos_w               # (N,3)
    base_quat_w = robot.data.root_quat_w             # (N,4)
    base_vel_w = robot.data.root_lin_vel_w           # (N,3)

    # forward direction (world)
    fwd_b = torch.tensor([1.0, 0.0, 0.0], device=device).view(1, 3).repeat(N, 1)
    fwd_w = _quat_rotate(base_quat_w, fwd_b)         # (N,3)

    v_fwd = (base_vel_w * fwd_w).sum(dim=-1)         # (N,) m/s
    v_fwd_pos = v_fwd.clamp(min=0.0)

    # -----------------------------
    # 3) reset mask
    # -----------------------------
    if hasattr(env, "episode_length_buf"):
        reset_mask = env.episode_length_buf == 0
    else:
        reset_mask = torch.zeros((N,), dtype=torch.bool, device=device)

    # -----------------------------
    # 4) gap_seen latch + start pose
    # -----------------------------
    if (not hasattr(env, "_td_gap_seen")) or (env._td_gap_seen.shape[0] != N):
        env._td_gap_seen = torch.zeros((N,), dtype=torch.bool, device=device)
    if (not hasattr(env, "_td_gap_start_pos")) or (env._td_gap_start_pos.shape[0] != N):
        env._td_gap_start_pos = base_pos_w.clone()
    if (not hasattr(env, "_td_gap_start_fwd")) or (env._td_gap_start_fwd.shape[0] != N):
        env._td_gap_start_fwd = fwd_w.clone()

    env._td_gap_seen = torch.where(reset_mask, torch.zeros_like(env._td_gap_seen), env._td_gap_seen)
    env._td_gap_start_pos = torch.where(reset_mask.unsqueeze(1), base_pos_w, env._td_gap_start_pos)
    env._td_gap_start_fwd = torch.where(reset_mask.unsqueeze(1), fwd_w, env._td_gap_start_fwd)

    new_gap = gap_present & (~env._td_gap_seen)
    env._td_gap_seen = env._td_gap_seen | gap_present
    gap_active = env._td_gap_seen

    env._td_gap_start_pos = torch.where(new_gap.unsqueeze(1), base_pos_w, env._td_gap_start_pos)
    env._td_gap_start_fwd = torch.where(new_gap.unsqueeze(1), fwd_w, env._td_gap_start_fwd)

    travel = ((base_pos_w - env._td_gap_start_pos) * env._td_gap_start_fwd).sum(dim=-1)  # (N,)

    # dtravel (per-step); convert to speed-like by /dt so weights are stable
    if (not hasattr(env, "_td_prev_travel")) or (env._td_prev_travel.shape[0] != N):
        env._td_prev_travel = travel.clone()
    env._td_prev_travel = torch.where(reset_mask, travel, env._td_prev_travel)

    dtravel = travel - env._td_prev_travel
    env._td_prev_travel = travel.clone()

    v_travel_pos = (dtravel.clamp(min=0.0) / max(dt, 1e-6))   # (N,) ~ m/s
    v_travel_neg = ((-dtravel).clamp(min=0.0) / max(dt, 1e-6))

    # -----------------------------
    # 5) feet -> patch index
    # -----------------------------
    feet_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :]  # (N,B,3)
    rel_w = feet_pos_w - base_pos_w.unsqueeze(1)
    rel_b = _quat_rotate_inv(base_quat_w.unsqueeze(1), rel_w)

    fx = rel_b[..., 0]
    fy = rel_b[..., 1]

    dy_cell = (y1 - y0) / (H - 1)
    ix = torch.round((fx - x0) / dx_cell).long()
    iy = torch.round((fy - y0) / dy_cell).long()

    in_patch = (
        (fx >= x0) & (fx <= x1) &
        (fy >= y0) & (fy <= y1) &
        (ix >= 0) & (ix < W) &
        (iy >= 0) & (iy < H)
    )

    ix_c = ix.clamp(0, W - 1)
    iy_c = iy.clamp(0, H - 1)
    flat_idx = (iy_c * W + ix_c)

    is_good_cell = is_good.view(N, -1).gather(1, flat_idx)
    is_gap_cell  = is_gap.view(N, -1).gather(1, flat_idx)

    # -----------------------------
    # 6) touchdown (contact sensor)
    # -----------------------------
    contact: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    forces = _get_contact_forces(contact)  # (N, bodies, 3)
    if hasattr(contact_sensor_cfg, "body_ids") and contact_sensor_cfg.body_ids is not None:
        forces = forces[:, contact_sensor_cfg.body_ids, :]
    fmag = torch.linalg.norm(forces, dim=-1)   # (N,B)
    in_contact = fmag > contact_force_th

    touchdown = in_contact
    hist = _get_contact_forces_history(contact)
    if hist is not None and hist.shape[1] >= 2:
        if hasattr(contact_sensor_cfg, "body_ids") and contact_sensor_cfg.body_ids is not None:
            hist = hist[:, :, contact_sensor_cfg.body_ids, :]
        prev_fmag = torch.linalg.norm(hist[:, -2, :, :], dim=-1)
        prev_contact = prev_fmag > contact_force_th
        touchdown = in_contact & (~prev_contact)

    # -----------------------------
    # 7) command gate
    # -----------------------------
    cmd = env.command_manager.get_command(command_name)
    cmd_x = cmd[:, 0]
    cmd_gate = cmd_x > min_cmd_x

    # -----------------------------
    # 8) touchdown reward (event): /dt to avoid dt dilution
    # -----------------------------
    good_td = (touchdown & in_patch & is_good_cell).float().sum(dim=1)
    gap_td  = (touchdown & in_patch & is_gap_cell ).float().sum(dim=1)

    moving_gate = cmd_gate & (v_fwd_pos > min_base_vx)
    r_td = (good_td - gap_td) * moving_gate.float() * gap_active.float()
    r_td = r_td / max(dt, 1e-6)

    # -----------------------------
    # 9) progress/back/stuck (rate): use v_travel to prevent shaking farm
    # -----------------------------
    r_prog  = progress_w * v_travel_pos * gap_active.float()
    r_back  = -back_w   * v_travel_neg * gap_active.float() * cmd_gate.float()
    r_stuck = -stuck_w  * ((v_travel_pos < 0.01).float()) * gap_active.float() * cmd_gate.float()

    # -----------------------------
    # 10) milestone jackpot for continuous stepping stones (event)
    # -----------------------------
    # --- 需要：在 new_gap 時也存一個 start_yaw，避免轉彎刷 ---
    # (放在你 gap_seen latch 初始化那一段附近)

    def _quat_to_yaw(q_wxyz: torch.Tensor) -> torch.Tensor:
        # q = (w,x,y,z)
        w, x, y, z = q_wxyz.unbind(-1)
        # yaw (z-axis)
        return torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

    def _wrap_pi(a: torch.Tensor) -> torch.Tensor:
        return (a + torch.pi) % (2*torch.pi) - torch.pi

    base_yaw = _quat_to_yaw(base_quat_w)  # (N,)
    if (not hasattr(env, "_td_gap_start_yaw")) or (env._td_gap_start_yaw.shape[0] != N):
        env._td_gap_start_yaw = base_yaw.clone()
    env._td_gap_start_yaw = torch.where(reset_mask, base_yaw, env._td_gap_start_yaw)
    env._td_gap_start_yaw = torch.where(new_gap, base_yaw, env._td_gap_start_yaw)

    yaw_dev = _wrap_pi(base_yaw - env._td_gap_start_yaw).abs()  # (N,)

    # --- (A) 定義「跨越落腳區」：必須落在前方 + 中線 ---
    cross_zone = (fx > cross_x_th) & (fy.abs() < cross_y_th) & in_patch  # (N,B)

    # touchdown 且落在 good cell 且在 cross zone
    cross_td = touchdown & cross_zone & is_good_cell  # (N,B)
    cross_td_any = cross_td.any(dim=1)                # (N,)

    # --- (B) anti-turn gate ---
    # cmd_y gate (base_velocity 通常是 [vx, vy, wz] 或 [vx, vy, yawrate]，你自己確認維度)
    cmd_y = cmd[:, 1] if cmd.shape[1] > 1 else torch.zeros_like(cmd_x)
    ang_w = robot.data.root_ang_vel_w  # (N,3) world
    yaw_rate = ang_w[:, 2].abs()

    straight_gate = (yaw_dev < yaw_dev_th) & (yaw_rate < yaw_rate_th) & (cmd_y.abs() < cmd_y_th)

    # --- (C) cooldown / 每顆 stone 只給一次：用 travel 當「進度尺」，但不拿來判定成功 ---
    if (not hasattr(env, "_td_last_bonus_travel")) or (env._td_last_bonus_travel.shape[0] != N):
        env._td_last_bonus_travel = travel.clone()
    env._td_last_bonus_travel = torch.where(reset_mask, travel, env._td_last_bonus_travel)

    enough_advance = (travel - env._td_last_bonus_travel) >= cross_dist  # (N,)

    hit_bonus = (
        gap_active &
        cmd_gate &
        straight_gate &
        (v_fwd > cross_min_vx) &
        (good_ratio > cross_good_ratio_th) &
        cross_td_any &
        enough_advance
    )

    r_bonus = (cross_bonus / max(dt, 1e-6)) * hit_bonus.float()
    env._td_last_bonus_travel = torch.where(hit_bonus, travel, env._td_last_bonus_travel)

    return r_td + r_prog + r_stuck + r_back + r_bonus

def commanded_standstill_penalty(
    env: ManagerBasedEnv,
    command_name: str = "base_velocity",
    min_cmd_x: float = 0.3,
    max_vx: float = 0.08,
) -> torch.Tensor:
    """
    If commanded forward but base_vx stays small -> penalty.
    """
    robot = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    cmd_x = cmd[:, 0]
    vx = robot.data.root_lin_vel_w[:, 0]
    return ((cmd_x > min_cmd_x) & (vx < max_vx)).float()
    
    
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

# 你如果已有這兩個 helper，沿用即可：
# - _quat_rotate_inv(q, v)
# - _get_contact_forces(contact)  -> (N, bodies, 3)
# - _get_contact_forces_history(contact) -> (N, T, bodies, 3) or None

import torch
from isaaclab.utils.math import quat_rotate_inverse
from isaaclab.utils.math import quat_apply_inverse  # ✅ 推薦

def _quat_rotate_inverse_batched(q_wxyz: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate v into frame of q^{-1}. Supports v shape (N,B,3) with q shape (N,4)."""
    # q: (N,4), v: (N,B,3)
    N, B = v.shape[0], v.shape[1]
    q_exp = q_wxyz.unsqueeze(1).expand(N, B, 4)         # (N,B,4)
    q_flat = q_exp.reshape(-1, 4)                       # (N*B,4)
    v_flat = v.reshape(-1, 3)                           # (N*B,3)
    out = quat_rotate_inverse(q_flat, v_flat)           # (N*B,3)
    return out.reshape(N, B, 3)                         # (N,B,3)

def hole_touchdown_penalty(
    env,
    height_sensor_cfg,
    contact_sensor_cfg,
    asset_cfg,                      # 用來拿 robot + foot body_ids
    grid_shape: tuple[int, int] = (11, 17),     # (Ny, Nx)
    size_xy: tuple[float, float] = (1.6, 1.0),  # (sx, sy)
    offset_z: float = 0.3,                      # 跟你 obs 一致
    hole_th: float = -0.05,                     # 平均高度 < hole_th 算踩洞
    power: float = 1.0,
    contact_force_th: float = 5.0,
    command_name: str = "base_velocity",
    min_cmd_x: float = 0.2,
    event_penalty: float = 5.0,                 # 一次 touchdown 事件的強度（再調大）
) -> torch.Tensor:
    dt = float(getattr(env, "step_dt", 0.02))

    # -----------------------------
    # 1) height grid (N, Ny, Nx)
    # -----------------------------
    sensor = env.scene.sensors[height_sensor_cfg.name]
    h = sensor.data.ray_hits_w[..., 2] - sensor.data.pos_w[:, 2].unsqueeze(1)  # (N, M)
    h = h + offset_z

    Ny, Nx = grid_shape
    N = h.shape[0]
    device = h.device

    if h.shape[1] != Ny * Nx:
        return torch.zeros((N,), device=device)

    h2 = h.view(N, Ny, Nx)            # (N,Ny,Nx)
    flat = h2.view(N, -1)             # (N,Ny*Nx)

    sx, sy = size_xy
    x0, x1 = -sx * 0.5, sx * 0.5
    y0, y1 = -sy * 0.5, sy * 0.5
    dx = sx / (Nx - 1)
    dy = sy / (Ny - 1)

    # -----------------------------
    # 2) feet positions -> base frame
    # -----------------------------
    robot = env.scene[asset_cfg.name]
    base_pos_w = robot.data.root_pos_w              # (N,3)
    base_quat_w = robot.data.root_quat_w            # (N,4)

    feet_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :]  # (N,B,3)
    rel_w = feet_pos_w - base_pos_w.unsqueeze(1)                  # (N,B,3)
    q = base_quat_w.unsqueeze(1).expand(-1, rel_w.shape[1], -1)  # (N,B,4)
    rel_b = quat_apply_inverse(q, rel_w)                         # (N,B,3)

    # scanner 平面座標
    fx = rel_b[..., 0] 
    fy = rel_b[..., 1]

    # in_patch：只在 scanner 覆蓋區域才判定
    in_patch = (fx >= x0) & (fx <= x1) & (fy >= y0) & (fy <= y1)

    # -----------------------------
    # 3) 取離腳最近的 3 個 grid 點（3NN）-> 平均高度
    # -----------------------------
    # 連續 index (u,v)
    u = (fx - x0) / dx   # x index float
    v = (fy - y0) / dy   # y index float

    # clamp 到可形成 2x2 cell 的範圍：ix0 in [0..Nx-2], iy0 in [0..Ny-2]
    u_c = u.clamp(0.0, (Nx - 1) - 1e-6)
    v_c = v.clamp(0.0, (Ny - 1) - 1e-6)
    ix0 = torch.floor(u_c).long().clamp(0, Nx - 2)
    iy0 = torch.floor(v_c).long().clamp(0, Ny - 2)

    # 2x2 四個角點候選（離腳最近一定在這四個）
    cand_ix = torch.stack([ix0, ix0 + 1, ix0, ix0 + 1], dim=2)   # (N,B,4)
    cand_iy = torch.stack([iy0, iy0, iy0 + 1, iy0 + 1], dim=2)   # (N,B,4)

    # 距離（在 index 空間算就夠了）
    u4 = u_c.unsqueeze(2).expand_as(cand_ix).float()
    v4 = v_c.unsqueeze(2).expand_as(cand_iy).float()
    dist2 = (u4 - cand_ix.float()) ** 2 + (v4 - cand_iy.float()) ** 2  # (N,B,4)

    # 取最近 3 點
    nn3 = dist2.topk(k=3, dim=2, largest=False).indices              # (N,B,3)
    cand_flat = (cand_iy * Nx + cand_ix)                              # (N,B,4)
    idx3 = torch.gather(cand_flat, 2, nn3)                            # (N,B,3)

    # gather heights：flat (N, Ny*Nx)
    h3 = flat.gather(1, idx3.reshape(N, -1)).reshape(N, -1, 3)        # (N,B,3)
    h_foot = h3.mean(dim=2)                                           # (N,B) ✅ 3NN 平均高度

    # -----------------------------
    # 4) touchdown (contact rising edge)
    # -----------------------------
    contact = env.scene.sensors[contact_sensor_cfg.name]

    # 這裡假設你原本已經有 _get_contact_forces / _get_contact_forces_history
    forces = _get_contact_forces(contact)  # (N, bodies, 3)

    if hasattr(contact_sensor_cfg, "body_ids") and contact_sensor_cfg.body_ids is not None:
        forces = forces[:, contact_sensor_cfg.body_ids, :]
    else:
        forces = forces[:, asset_cfg.body_ids, :]

    fmag = torch.linalg.norm(forces, dim=-1)  # (N,B)
    in_contact_now = fmag > contact_force_th

    touchdown = in_contact_now
    hist = _get_contact_forces_history(contact)
    if hist is not None and hist.shape[1] >= 2:
        if hasattr(contact_sensor_cfg, "body_ids") and contact_sensor_cfg.body_ids is not None:
            hist = hist[:, :, contact_sensor_cfg.body_ids, :]
        else:
            hist = hist[:, :, asset_cfg.body_ids, :]
        prev_fmag = torch.linalg.norm(hist[:, -2, :, :], dim=-1)  # (N,B)
        in_contact_prev = prev_fmag > contact_force_th
        touchdown = in_contact_now & (~in_contact_prev)

    # -----------------------------
    # 5) command gate
    # -----------------------------
    cmd = env.command_manager.get_command(command_name)
    cmd_gate = (cmd[:, 0] > min_cmd_x)

    # -----------------------------
    # 6) event penalty：touchdown 且平均高度落洞才扣
    # -----------------------------
    depth = (hole_th - h_foot).clamp(min=0.0)                 # (N,B)
    hole_td = touchdown & in_patch & (depth > 0.0)            # (N,B)

    # “平均起來多負扣多少”：用 depth^power
    pen = ((depth ** power) * hole_td.float()).sum(dim=1)     # (N,)

    # event：除以 dt，讓 RewardManager 乘回 dt 後 ≈ event_penalty * depth
    r = -event_penalty * pen / max(dt, 1e-6)

    r = torch.where(cmd_gate, r, torch.zeros_like(r))
    return r


