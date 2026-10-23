import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


class WalkingPose:
    def __init__(self, config, rng):
        data = json.loads((Path(__file__).parent / "data" / "walking.json").read_text(encoding="utf-8"))
        self.config = config
        self.rng = rng
        self.dt = data["dt"]
        self.scale = np.array(data["scale"])
        self.waves = np.array([clip["wave"] for clip in data["clips"]])
        self.waves[:, :, :3] -= self.waves[:, :, :3].mean(axis=1, keepdims=True)
        self.waves[:, :, 5] -= self.waves[:, :, 5].mean(axis=1, keepdims=True)
        velocities = np.array([clip["condition"] for clip in data["clips"]])
        self.velocities = np.repeat(velocities, 2, axis=0)
        self.velocities[1::2, 1:] *= -1
        self.durations = np.repeat([clip["duration"] for clip in data["clips"]], 2)
        self.phase = 0.0
        self.velocity = np.zeros(3)
        self.noise = np.zeros(6)
        self.position = np.zeros(3)
        self.rotation = np.eye(3)
        self.minimum = np.array(data["minimum"])
        self.maximum = np.array(data["maximum"])
        self.linear_velocity = np.zeros(3)
        self.yaw_rate = 0.0
        self.render()

    def weights(self):
        distance = np.sum(((self.velocities - self.velocity) / self.scale) ** 2, axis=1)
        exact = distance < 1e-14
        if exact.any():
            return exact.astype(float) / exact.sum()
        weights = 1.0 / np.maximum(distance, 1e-14) ** 2
        weights /= weights.sum()
        weights[weights < 1e-6] = 0.0
        return weights / weights.sum()

    def advance(self, command, duration=None):
        duration = self.config.dt if duration is None else duration
        if not np.isfinite(duration) or duration < 0:
            raise ValueError("Walking duration must be finite and nonnegative")
        substeps = int(math.ceil(duration / self.dt - 1e-10))
        target = np.clip(command, self.minimum, self.maximum)
        deviation = np.array([self.config.gait_noise_position] * 3 + [self.config.gait_noise_angle] * 3)
        for index in range(substeps):
            dt = min(self.dt, duration - index * self.dt)
            response = 1.0 - math.exp(-dt / 0.18)
            correlation = math.exp(-dt / self.config.gait_noise_tau)
            self.velocity += response * (target - self.velocity)
            weights = self.weights()
            self.phase = (self.phase + dt * np.sum(weights / self.durations)) % 1.0
            self.noise = correlation * self.noise + math.sqrt(1.0 - correlation ** 2) * deviation * self.rng.normal(size=6)
            previous_position, previous_rotation = self.position.copy(), self.rotation.copy()
            self.render()
            self.linear_velocity = (self.position - previous_position) / dt
            self.yaw_rate = Rotation.from_matrix(self.rotation @ previous_rotation.T).as_rotvec()[2] / dt
        return self.position.copy(), self.rotation.copy()

    def render(self):
        weights = self.weights()
        phases = (self.phase + np.tile([0.0, 0.5], len(self.waves))) % 1.0
        samples = phases * self.waves.shape[1]
        left = np.floor(samples).astype(int)
        right = (left + 1) % self.waves.shape[1]
        fraction = samples - left
        indices = np.arange(len(weights)) // 2
        values = self.waves[indices, left] * (1.0 - fraction[:, None]) + self.waves[indices, right] * fraction[:, None]
        values[1::2, 1] *= -1
        values[1::2, 3] *= -1
        values[1::2, 5] *= -1
        wave = weights @ values + self.noise
        self.position = wave[:3]
        self.rotation = Rotation.from_euler("xyz", wave[3:]).as_matrix()
