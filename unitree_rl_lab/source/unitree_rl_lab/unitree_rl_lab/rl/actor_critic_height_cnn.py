from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple, Literal

import torch
import torch.nn as nn
from torch.distributions import Normal

try:
    from tensordict import TensorDictBase
except Exception:
    TensorDictBase = tuple()
    
def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name in ("lrelu", "leakyrelu"):
        return nn.LeakyReLU(0.2)
    raise ValueError(f"Unsupported activation: {name}")


def _build_mlp(in_dim: int, hidden_dims: Sequence[int], out_dim: int, activation: str) -> nn.Sequential:
    act = _get_activation(activation)
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(last, h), act]
        last = h
    layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)
    
def _build_mlp_lazy(hidden_dims: Sequence[int], out_dim: int, activation: str) -> nn.Sequential:
    act = _get_activation(activation)
    layers: list[nn.Module] = []
    # 第一層用 LazyLinear，不用知道 in_dim
    layers += [nn.LazyLinear(hidden_dims[0]), act]
    last = hidden_dims[0]
    for h in hidden_dims[1:]:
        layers += [nn.Linear(last, h), act]
        last = h
    layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)
    
def _resolve_obs_dim(x, preferred_keys=("policy", "critic", "obs", "actor", "value")) -> int:
    # int
    if isinstance(x, int):
        return x

    # torch tensor -> last dim
    if torch.is_tensor(x):
        return int(x.shape[-1])

    # tensordict -> pick preferred key then recurse
    if TensorDictBase and isinstance(x, TensorDictBase):
        for k in preferred_keys:
            if k in x.keys():
                return _resolve_obs_dim(x.get(k), preferred_keys=preferred_keys)
        # if only one key, recurse into it
        keys = list(x.keys())
        if len(keys) == 1:
            return _resolve_obs_dim(x.get(keys[0]), preferred_keys=preferred_keys)
        raise TypeError(f"Cannot resolve obs dim from TensorDict keys={keys}")

    # plain dict -> similar logic
    if isinstance(x, dict):
        for k in preferred_keys:
            if k in x:
                return _resolve_obs_dim(x[k], preferred_keys=preferred_keys)
        if len(x) == 1:
            return _resolve_obs_dim(next(iter(x.values())), preferred_keys=preferred_keys)
        raise TypeError(f"Cannot resolve obs dim from dict: {list(x.keys())}")

        # list/tuple: multiple observation parts -> total dim is sum
    if isinstance(x, (list, tuple)):
        return int(sum(_resolve_obs_dim(v, preferred_keys=preferred_keys) for v in x))

    raise TypeError(f"Unsupported obs dim type: {type(x)}")
    
def _get_dim_from_obs_sample(obs_sample, mode: str) -> int:
    """
    obs_sample can be:
      - torch.Tensor: (B, D)
      - dict: {"policy": tensor, "critic": tensor} or {"policy":[...], ...}
      - TensorDict
    """
    # choose key
    key = "policy" if mode == "actor" else "critic"

    # tensor
    if torch.is_tensor(obs_sample):
        return int(obs_sample.shape[-1])

    # dict
    if isinstance(obs_sample, dict):
        x = obs_sample.get(key, None)
        if x is None:
            # fallback: take first value
            x = next(iter(obs_sample.values()))
        return _get_dim_from_obs_sample(x, mode=mode) if not torch.is_tensor(x) else int(x.shape[-1])

    # tensordict
    if TensorDictBase and isinstance(obs_sample, TensorDictBase):
        if key in obs_sample.keys():
            return _get_dim_from_obs_sample(obs_sample.get(key), mode=mode)
        # fallback: pick something
        keys = list(obs_sample.keys())
        return _get_dim_from_obs_sample(obs_sample.get(keys[0]), mode=mode)

    # list/tuple -> concat dims
    if isinstance(obs_sample, (list, tuple)):
        # list items should be tensors or nested structures
        dims = []
        for v in obs_sample:
            if isinstance(v, str):
                # this is a group name, skip (not actual tensor)
                continue
            dims.append(_get_dim_from_obs_sample(v, mode=mode))
        return int(sum(dims))

    # if it's a string (group name), we cannot resolve from it
    if isinstance(obs_sample, str):
        raise TypeError(f"obs_sample is a string '{obs_sample}', expected tensor sample.")

    raise TypeError(f"Unsupported obs_sample type: {type(obs_sample)}")


class HeightCNNEncoder(nn.Module):
    """Encode height-scan grid (H*W) -> latent vector."""

    def __init__(self, hw: Tuple[int, int], out_dim: int = 64, activation: str = "elu"):
        super().__init__()
        self.h, self.w = hw
        act = _get_activation(activation)

        # (B, 1, H, W) -> feature map
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            act,
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            act,
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            act,
        )

        # 推一次 dummy 來算 flatten 維度（避免你換 H,W 就要手算）
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.h, self.w)
            flat_dim = int(self.cnn(dummy).flatten(1).shape[1])

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, out_dim),
            act,
        )

    def forward(self, height_flat: torch.Tensor) -> torch.Tensor:
        # height_flat: (B, H*W)
        b = height_flat.shape[0]
        x = height_flat.view(b, 1, self.h, self.w)
        x = self.cnn(x)
        z = self.head(x)
        return z  # (B, out_dim)


class ActorCriticHeightCNN(nn.Module):
    """
    Actor-Critic with a CNN encoder for the height_scanner part of the observation.

    Assumption (matches IsaacLab ObsGroup concatenate_terms=True):
    - actor obs is a flat vector of length num_actor_obs
    - height scan is the LAST `height_scan_dim` entries of that vector
      (same for critic obs)
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: Sequence[int] = (512, 256, 128),
        critic_hidden_dims: Sequence[int] = (512, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: Literal["scalar", "log"] = "scalar",
        actor_obs_normalization: bool | None = None,   # runner 可能會丟，不用也沒關係
        critic_obs_normalization: bool | None = None,  # runner 可能會丟，不用也沒關係
        # --- height scan specific ---
        height_scan_dim: int = 187,
        height_scan_hw: Tuple[int, int] = (17, 11),
        height_latent_dim: int = 64,
        obs=None,
        entropy = None,
        **kwargs,   # ✅ 接住其他你沒列到的參數，避免再噴 unexpected keyword
    ):
        super().__init__()

        if height_scan_dim != height_scan_hw[0] * height_scan_hw[1]:
            raise ValueError(
                f"height_scan_dim ({height_scan_dim}) != H*W ({height_scan_hw[0]}*{height_scan_hw[1]})"
            )

        self.num_actions = num_actions
        self.height_scan_dim = height_scan_dim
        self.height_scan_hw = height_scan_hw
        self.height_latent_dim = height_latent_dim

        # --- encoders ---
        self.height_encoder_actor = HeightCNNEncoder(height_scan_hw, out_dim=height_latent_dim, activation=activation)
        self.height_encoder_critic = HeightCNNEncoder(height_scan_hw, out_dim=height_latent_dim, activation=activation)
        
        # --- MLPs (lazy) ---
        self.actor = _build_mlp_lazy(actor_hidden_dims, num_actions, activation)
        self.critic = _build_mlp_lazy(critic_hidden_dims, 1, activation)

        # --- action distribution ---
        self.noise_std_type = noise_std_type

        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            self.log_std = None
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            self.std = None
        else:
            raise ValueError(f"Unsupported noise_std_type: {noise_std_type}")
        self._dist: Normal | None = None

    # ---------------------------
    # internal: feature builders
    # ---------------------------
    def _actor_features(self, obs) -> torch.Tensor:
        obs = self._unwrap_obs(obs, "actor")
        proprio = obs[:, : -self.height_scan_dim]
        height = obs[:, -self.height_scan_dim :]
        z = self.height_encoder_actor(height)
        return torch.cat([proprio, z], dim=-1)


    def _critic_features(self, obs) -> torch.Tensor:
        obs = self._unwrap_obs(obs, "critic")
        proprio = obs[:, : -self.height_scan_dim]
        height = obs[:, -self.height_scan_dim :]
        z = self.height_encoder_critic(height)
        return torch.cat([proprio, z], dim=-1)

    def _force_non_inference(self, t: torch.Tensor) -> torch.Tensor:
        # 最保險：只要是 inference tensor 就 clone 成一般 tensor
        try:
            if torch.is_tensor(t) and t.is_inference():
                return t.clone()
        except Exception:
            pass
        return t

    def _to_normal_tensor(self, t: torch.Tensor) -> torch.Tensor:
        # 必須在 inference_mode(False) 內呼叫這個 clone 才會變成 normal tensor
        try:
            if t.is_inference():
                return t.clone()
        except Exception:
            pass
        return t

    def _current_std(self, mean: torch.Tensor) -> torch.Tensor:
        if self.noise_std_type == "scalar":
            return mean * 0.0 + self.std
        else:
            return mean * 0.0 + torch.exp(self.log_std)
    def update_distribution(self, obs, **kwargs) -> None:
        # ✅ 強制關掉 inference_mode，否則 backward 會因 inference tensor 爆炸
        with torch.inference_mode(False):
            obs_t = self._unwrap_obs(obs, "actor")     # 可能是 inference tensor
            obs_t = self._to_normal_tensor(obs_t)      # 在 inference_mode(False) 內 clone -> 變 normal

            feat = self._actor_features_from_tensor(obs_t)
            mean = self.actor(feat)

            std = self._current_std(mean)

            # rsl_rl PPO needs these
            self.action_mean = mean
            self.action_std = std
            self._dist = Normal(mean, std)
            # ✅ rsl_rl expects entropy as a tensor attribute, not a method
            self.entropy = self._dist.entropy().sum(dim=-1)


    # ---------------------------
    # API used by rsl_rl
    # ---------------------------
    def act(self, obs, **kwargs) -> torch.Tensor:
        # kwargs may include masks, hidden_state, etc. (ignored for non-recurrent policy)
        self.update_distribution(obs, **kwargs)
        return self._dist.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        assert self._dist is not None
        return self._dist.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs, **kwargs) -> torch.Tensor:
        return self.actor(self._actor_features(obs))

    def evaluate(self, critic_obs, **kwargs) -> torch.Tensor:
        with torch.inference_mode(False):
            obs_t = self._unwrap_obs(critic_obs, "critic")
            obs_t = self._to_normal_tensor(obs_t)

            feat = self._critic_features_from_tensor(obs_t)
            v = self.critic(feat)
            return v  # (N,1) 不要 squeeze

    def _actor_features_from_tensor(self, obs_t: torch.Tensor) -> torch.Tensor:
        proprio = obs_t[:, : -self.height_scan_dim]
        height  = obs_t[:, -self.height_scan_dim :]
        z = self.height_encoder_actor(height)
        return torch.cat([proprio, z], dim=-1)

    def _critic_features_from_tensor(self, obs_t: torch.Tensor) -> torch.Tensor:
        proprio = obs_t[:, : -self.height_scan_dim]
        height  = obs_t[:, -self.height_scan_dim :]
        z = self.height_encoder_critic(height)
        return torch.cat([proprio, z], dim=-1)


    def reset(self, dones: torch.Tensor | None = None) -> None:
        # non-recurrent: nothing to reset
        return
    def update_normalization(self, obs) -> None:
        """Called by rsl_rl PPO each env step to update obs normalizers.
        We currently do not use an explicit normalizer inside this custom policy.
        """
        return
    def _to_trainable_tensor(self, t: torch.Tensor) -> torch.Tensor:
        # If it's an inference tensor, clone it to make it a normal tensor for autograd.
        try:
            if t.is_inference():
                return t.clone()
        except Exception:
            pass
        return t

    def _ensure_tensor_obs(self, x):
        # already tensor
        if torch.is_tensor(x):
            return self._to_trainable_tensor(x)
    
        # list/tuple of tensors (or nested structures) -> concat
        if isinstance(x, (list, tuple)):
            parts = [self._ensure_tensor_obs(v) for v in x]
            return torch.cat(parts, dim=-1)

        raise TypeError(f"Cannot convert obs part to tensor, type={type(x)}")
        
    def _unwrap_obs(self, obs, mode: str):
        if torch.is_tensor(obs):
            return self._ensure_tensor_obs(obs)
    
        if isinstance(obs, dict):
            key = "policy" if mode == "actor" else "critic"
            if key in obs:
                return self._ensure_tensor_obs(obs[key])
            for k in ("obs", "actor", "value", "policy", "critic"):
                if k in obs:
                    return self._ensure_tensor_obs(obs[k])

        if TensorDictBase and isinstance(obs, TensorDictBase):
            key = "policy" if mode == "actor" else "critic"
            if key in obs.keys():
                return self._ensure_tensor_obs(obs.get(key))
            for k in ("obs", "actor", "value", "policy", "critic"):
                if k in obs.keys():
                    return self._ensure_tensor_obs(obs.get(k))
    
        raise TypeError(f"Cannot unwrap obs for mode={mode}, type={type(obs)}")


