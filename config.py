from dataclasses import dataclass
from math import radians
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass
class TaskConfig:
    dt: float = 0.1
    physics_dt: float = 0.003
    uav_speed: float = 4.0
    ugv_speed: float = 1.0
    ugv_yaw_rate: float = radians(90)
    nominal_height: float = 1.2
    minimum_height: float = 0.5
    maximum_height: float = 1.2
    posture_limit: float = radians(20)
    residual: tuple = (1.5, 1.0, radians(20), 0.6, radians(20), radians(20))
    platform_half_size: tuple = (0.25, 0.25)
    platform_offset: tuple = (0.0, 0.0, 0.0)
    gear_offset: tuple = (0.0, 0.0, 0.0)
    ue_origin: tuple = (0.0, 0.0, 0.0)
    ue_heading: float = 0.0
    normal_speed_limit: float = 1.0
    tangent_speed_limit: float = 1.0
    attitude_limit: float = radians(15)
    safety_radius: float = 23.0
    episode_steps: int = 150
    marker_loss_steps: int = 10
    waypoint_radius: float = 0.5
    wind_speed: float = 3.0
    wind_interval: int = 5
    sensor_noise: float = 0.05
    latency_ms: float = 0.0
    packet_loss: float = 0.0
    gait_noise_position: float = 0.0005
    gait_noise_angle: float = radians(0.1)
    gait_noise_tau: float = 0.2
    kf_horizon: int = 5
    kf_acceleration_std: float = 1.0
    kf_measurement_std: float = 0.1
    host: str = "127.0.0.1"
    send_port: int = 8000
    listen_port: int = 9000
    state_timeout: float = 2.0
    step_timeout: float = 10.0
    external_marker: bool = False
    velocity_response: tuple = (0.5, 0.5, 0.5)
    expert_min_duration: float = 0.03
    expert_max_rmse: float = 0.2
    expert_regularization: float = 0.001
    platform_object: str = ""
    contact_tolerance: float = 0.08
    contact_normal_cosine: float = 0.9
    yaw_kp: float = 1.0
    yaw_ki: float = 0.1
    yaw_kd: float = 0.05
    yaw_limit: float = radians(60)


@dataclass
class TrainConfig:
    seed: int = 43342521
    steps: int = 150000
    phase1_end: int = 25000
    phase2_end: int = 50000
    batch_size: int = 512
    replay_size: int = 50000
    hidden_dim: int = 512
    actor_lr: float = 5e-5
    critic_lr: float = 1e-4
    alpha_lr: float = 2e-5
    gamma: float = 0.98
    tau: float = 0.005
    alpha: float = 1.0
    target_entropies: tuple = (-3.0, -6.0)
    expert_ratio_init: float = 0.3
    expert_ratio: float = 0.5
    bc_weight: float = 1.0
    decay_steps: int = 100000
    redistribution_steps: int = 10
    redistribution_decay: float = 0.5
    save_interval: int = 10000

    def phase(self, step):
        return 1 if step < self.phase1_end else 2 if step < self.phase2_end else 3

    def guidance(self, step, phase):
        if phase != 1:
            return 0.0, 0.0
        if step is None:
            return self.expert_ratio, self.bc_weight
        progress = max(0.0, 1.0 - step / self.decay_steps)
        return self.expert_ratio_init * progress, self.bc_weight * progress
