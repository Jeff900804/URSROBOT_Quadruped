from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase
    
from isaaclab.sensors import RayCaster
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedEnv

def height_scan(env, sensor_cfg, offset=0.5):
    sensor = env.scene.sensors[sensor_cfg.name]
    # 反號：hit 在更下面 => 更負
    return sensor.data.ray_hits_w[..., 2] - sensor.data.pos_w[:, 2].unsqueeze(1) + offset
# -------------------------
# Quaternion helpers (wxyz)
# -------------------------
def _quat_conj(q: torch.Tensor) -> torch.Tensor:
    # q: (..., 4) wxyz
    return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)

def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a,b: (...,4) wxyz
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack([w, x, y, z], dim=-1)

def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # rotate v by q (wxyz). v: (...,3)
    qv = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
    return _quat_mul(_quat_mul(q, qv), _quat_conj(q))[..., 1:4]

def _quat_rotate_inv(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # rotate v by inverse(q)
    return _quat_rotate(_quat_conj(q), v)

def _dbg_grid_stats(env, tag: str, grid: torch.Tensor, fill_value: float, every: int = 200) -> None:
    """Print grid stats for env0 every `every` steps."""
    # env0 only, throttled
    if not hasattr(env, "_dbg_counter"):
        env._dbg_counter_obs = 0
    env._dbg_counter_obs += 1
    if env._dbg_counter_obs % every != 0:
        return

    g0 = grid[1]  # env0
    fill_ratio = (g0 == fill_value).float().mean().item()
    print(
        f"[DBG][{tag}] grid0 mean/std/min/max = "
        f"{g0.mean().item():.6f} {g0.std().item():.6f} {g0.min().item():.6f} {g0.max().item():.6f} | "
        f"fill_ratio={fill_ratio:.3f}"
    )

# ---------------------------------------------------------
# 81-dim front patch from RayCaster (order-independent)
# ---------------------------------------------------------
def height_scan_front_patch(env, sensor_cfg, offset=0.5, x_range=(0.0,0.8), y_range=(-0.4,0.4),
                            grid_shape=(9,9), fill_value=-10.0):
    sensor = env.scene.sensors[sensor_cfg.name]

    # 這個定義：平面≈0，坑洞更負（你想要的符號）
    heights = sensor.data.ray_hits_w[..., 2] - sensor.data.pos_w[:, 2].unsqueeze(1) + offset  # (N,R)

    H, W = grid_shape
    if heights.shape[1] != H * W:
        raise RuntimeError(f"[height_scan_front_patch] Expect {H*W} rays, got {heights.shape[1]}. "
                           "Your pattern_cfg is not 9x9; then you must implement binning properly.")

    # 直接回傳 81 維（順序就跟 GridPattern 產生 ray 的順序一致）
    #_dbg_grid_stats(env, "OBS/front_patch", heights, fill_value, every=1)
    return heights    
