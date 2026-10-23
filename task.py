from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial.transform import Rotation


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def normalize(value, bound):
    return np.clip(np.asarray(value) / bound, -1.0, 1.0)


def spherical_velocity(action, maximum=4.0):
    radius, azimuth, elevation = np.clip(action, -1.0, 1.0)
    speed = maximum * (radius + 1.0) / 2.0
    azimuth *= math.pi
    elevation *= math.pi / 2.0
    return speed * np.array([math.cos(elevation) * math.cos(azimuth),
                             math.cos(elevation) * math.sin(azimuth), math.sin(elevation)])


def velocity_action(velocity, maximum=4.0):
    velocity = np.asarray(velocity, dtype=float)
    if velocity.shape != (3,) or not np.isfinite(velocity).all() or maximum <= 0:
        raise ValueError("Velocity must contain three finite values and a positive limit")
    speed = float(np.linalg.norm(velocity))
    if speed > maximum + 1e-6:
        raise ValueError("Velocity command exceeds the UAV speed limit")
    if speed < 1e-8:
        return np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    return np.array([2.0 * min(speed / maximum, 1.0) - 1.0,
                     math.atan2(velocity[1], velocity[0]) / math.pi,
                     math.asin(np.clip(velocity[2] / speed, -1.0, 1.0)) * 2.0 / math.pi], dtype=np.float32)


def infer_action(times, velocities, previous_command, config):
    times = np.asarray(times, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    invalid = {"valid": False, "rmse": None}
    if (times.ndim != 1 or len(times) < 4 or velocities.shape != (len(times), 3)
            or not np.isfinite(times).all() or not np.isfinite(velocities).all()
            or abs(times[0]) > 1e-9 or np.any(np.diff(times) <= 0)
            or times[-1] < config.expert_min_duration):
        return np.zeros(3, dtype=np.float32), invalid
    response = np.asarray(config.velocity_response, dtype=float)
    if response.shape != (3,) or not np.isfinite(response).all() or np.any((response <= 0.0) | (response >= 1.0)):
        raise ValueError("Velocity response coefficients must lie strictly between zero and one")
    decay = (1.0 - response) ** (times[1:, None] / config.dt)
    design = 1.0 - decay
    target = velocities[1:] - decay * velocities[0]
    weights = np.ones_like(target)
    regularization = config.expert_regularization
    for _ in range(4):
        command = (np.sum(weights * design * target, axis=0) + regularization * previous_command) / (np.sum(weights * design ** 2, axis=0) + regularization)
        residual = target - design * command
        weights = np.minimum(1.0, 0.1 / np.maximum(np.abs(residual), 1e-9))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    valid = bool(np.isfinite(command).all() and np.linalg.norm(command) <= config.uav_speed + 1e-6
                 and np.max(np.linalg.norm(velocities, axis=1)) <= config.uav_speed + 1e-6
                 and rmse <= config.expert_max_rmse)
    action = velocity_action(command, config.uav_speed) if valid else np.zeros(3, dtype=np.float32)
    return action, {"valid": valid, "rmse": rmse}


@dataclass
class State:
    position: np.ndarray
    rotation: np.ndarray
    velocity: np.ndarray
    omega: np.ndarray
    body_velocity: np.ndarray
    height: float

    @property
    def rpy(self):
        return Rotation.from_matrix(self.rotation).as_euler("xyz")


class Route:
    def __init__(self, points, speeds, config):
        self.points = np.vstack([np.zeros(2), np.asarray(points, dtype=float)])
        self.speeds = np.asarray(speeds, dtype=float)
        self.config = config
        self.index = 0
        self.integral = 0.0
        self.previous_error = 0.0

    @classmethod
    def generate(cls, rng, config, count=None):
        count = int(rng.integers(2, 5)) if count is None else count
        if count not in (2, 3, 4):
            raise ValueError("Waypoint count must be 2, 3, or 4")
        lengths = rng.uniform(max(3.0, 12.0 / count), min(7.0, 15.0 / count), count)
        headings = np.cumsum(rng.uniform(-math.pi / 3.0, math.pi / 3.0, count))
        vectors = lengths[:, None] * np.column_stack([np.cos(headings), np.sin(headings)])
        return cls(np.cumsum(vectors, axis=0), rng.uniform(0.6, 1.0, count), config)

    @property
    def finished(self):
        return self.index >= len(self.speeds)

    @property
    def length(self):
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())

    def advance(self, position):
        while not self.finished and np.linalg.norm(self.points[self.index + 1] - position[:2]) <= self.config.waypoint_radius:
            self.index += 1
            self.integral = 0.0
            self.previous_error = 0.0

    def geometry(self, position, yaw):
        index = min(self.index, len(self.speeds) - 1)
        start, end = self.points[index:index + 2]
        tangent = (end - start) / np.linalg.norm(end - start)
        delta = end - position[:2]
        direction = delta / max(np.linalg.norm(delta), 1e-9)
        offset = position[:2] - start
        lateral = tangent[0] * offset[1] - tangent[1] * offset[0]
        heading = wrap(yaw - math.atan2(tangent[1], tangent[0]))
        return direction, tangent, float(lateral), heading

    def command(self, position, yaw):
        cfg = self.config
        if self.finished:
            return np.array([0.0, 0.0, 0.0, cfg.nominal_height, 0.0, 0.0])
        delta = self.points[self.index + 1] - position[:2]
        distance = np.linalg.norm(delta)
        error = wrap(math.atan2(delta[1], delta[0]) - yaw)
        self.integral = float(np.clip(self.integral + error * cfg.dt, -math.pi / 3, math.pi / 3))
        derivative = wrap(error - self.previous_error) / cfg.dt
        self.previous_error = error
        yaw_rate = np.clip(0.6 * error + 0.05 * self.integral + 0.08 * derivative, -math.pi / 6, math.pi / 6)
        fraction = np.clip((distance - cfg.waypoint_radius) / (2.0 - cfg.waypoint_radius), 0.0, 1.0)
        speed = 0.2 + fraction * (self.speeds[self.index] - 0.2)
        return np.array([speed, 0.0, yaw_rate, cfg.nominal_height, 0.0, 0.0])

    def progress(self, before, after):
        if self.finished:
            return 0.0
        start, end = self.points[self.index:self.index + 2]
        length = np.linalg.norm(end - start)
        tangent = (end - start) / length
        return float(np.clip(np.dot(after[:2] - start, tangent), 0.0, length)
                     - np.clip(np.dot(before[:2] - start, tangent), 0.0, length))

    def observation(self, state, nominal):
        direction, tangent, lateral, heading = self.geometry(state.position, state.rpy[2])
        bounds = np.array([self.config.ugv_speed, self.config.ugv_speed, self.config.ugv_yaw_rate,
                           self.config.maximum_height, self.config.posture_limit, self.config.posture_limit])
        return np.concatenate([direction, tangent, normalize([lateral, heading], [23.0, math.pi]), normalize(nominal, bounds)])


class RelativeKF:
    def __init__(self, position, velocity, config, timestamp=0.0):
        self.config = config
        self.x = np.r_[position, velocity].astype(float)
        self.P = np.diag([config.kf_measurement_std ** 2] * 2 + [1.0] * 2)
        self.A, self.Q = self.matrices(config.dt)
        self.H = np.eye(4)[:2]
        self.R = np.eye(2) * config.kf_measurement_std ** 2
        self.time = float(timestamp)
        self.posterior_time = self.time
        self.posterior_x = self.x.copy()
        self.posterior_P = self.P.copy()

    def matrices(self, dt):
        transition = np.eye(4)
        transition[:2, 2:] = np.eye(2) * dt
        g = np.vstack([np.eye(2) * dt ** 2 / 2.0, np.eye(2) * dt])
        return transition, self.config.kf_acceleration_std ** 2 * g @ g.T

    def propagate(self, state, covariance, duration):
        state, covariance = state.copy(), covariance.copy()
        while duration > 1e-10:
            dt = min(self.config.dt, duration)
            transition, noise = self.matrices(dt)
            state = transition @ state
            covariance = transition @ covariance @ transition.T + noise
            duration -= dt
        return state, covariance

    def update(self, position=None, duration=None, measurement_time=None):
        duration = self.config.dt if duration is None else duration
        if not np.isfinite(duration) or duration < 0:
            raise ValueError("KF duration must be finite and nonnegative")
        self.time += duration
        stamp = self.time if measurement_time is None else float(measurement_time)
        corrected = position is not None and self.posterior_time + 1e-9 < stamp <= self.time + 1e-9
        if corrected:
            state, covariance = self.propagate(self.posterior_x, self.posterior_P, stamp - self.posterior_time)
            innovation = np.asarray(position) - self.H @ state
            gain = np.linalg.solve(self.H @ covariance @ self.H.T + self.R, self.H @ covariance).T
            self.posterior_x = state + gain @ innovation
            correction = np.eye(4) - gain @ self.H
            self.posterior_P = correction @ covariance @ correction.T + gain @ self.R @ gain.T
            self.posterior_time = stamp
        self.x, self.P = self.propagate(self.posterior_x, self.posterior_P, self.time - self.posterior_time)
        return corrected

    def predictions(self):
        times = np.arange(1, self.config.kf_horizon + 1) * self.config.dt
        return self.x[:2] + times[:, None] * self.x[2:]


def predict_state(state, duration):
    rotation = Rotation.from_rotvec(state.omega * duration).as_matrix() @ state.rotation
    return State(state.position + state.velocity * duration, rotation, state.velocity.copy(),
                 state.omega.copy(), state.body_velocity.copy(), state.height + state.velocity[2] * duration)


def contact_metrics(uav, ugv, config, impact=None):
    error = uav.position - ugv.position
    local = ugv.rotation.T @ error
    normal = ugv.rotation[:, 2]
    if impact is None:
        uav_contact = uav.position + uav.rotation @ np.asarray(config.gear_offset)
        ugv_contact = uav_contact - normal * np.dot(uav_contact - ugv.position, normal)
    else:
        uav_contact = ugv_contact = np.asarray(impact)
    relative_velocity = (uav.velocity + np.cross(uav.omega, uav_contact - uav.position)
                         - ugv.velocity - np.cross(ugv.omega, ugv_contact - ugv.position))
    normal_velocity = float(np.dot(relative_velocity, normal))
    tangent_velocity = relative_velocity - normal_velocity * normal
    angle = math.acos(float(np.clip(np.dot(uav.rotation[:, 2], normal), -1.0, 1.0)))
    metrics = {"distance": float(np.linalg.norm(error)), "error_x": float(abs(local[0])),
               "error_y": float(abs(local[1])), "error_xy": float(np.linalg.norm(local[:2])),
               "clearance": float(local[2]), "normal_speed": abs(normal_velocity),
               "tangent_speed": float(np.linalg.norm(tangent_velocity)), "attitude_error": angle}
    metrics["safe"] = bool(metrics["error_x"] <= config.platform_half_size[0]
                           and metrics["error_y"] <= config.platform_half_size[1]
                           and metrics["normal_speed"] <= config.normal_speed_limit
                           and metrics["tangent_speed"] <= config.tangent_speed_limit
                           and angle <= config.attitude_limit)
    return metrics


def team_reward(previous_distance, metrics, progress, nominal_progress, lateral, heading,
                uav_action, ugv_action, previous_uav_action, previous_ugv_action, event):
    dock = 10.0 * (previous_distance - metrics["distance"])
    route = 0.7 * progress - 0.3 * lateral ** 2 - 0.4 * heading ** 2 - 0.3 * max(0.0, nominal_progress - progress)
    smooth = (-0.5 - 0.02 * float(np.sum((uav_action - previous_uav_action) ** 2))
              - 0.2 * float(np.sum((ugv_action - previous_ugv_action) ** 2)))
    target = (148.0 - 80.0 * metrics["error_xy"] - 15.0 * max(0.0, metrics["normal_speed"] - 1.0)
              - 10.0 * max(0.0, metrics["tangent_speed"] - 1.0)
              - 200.0 * min(metrics["attitude_error"] / (math.pi / 2.0), 1.0)) if event == "touchdown" else 0.0
    terminal = target if event == "touchdown" else {"crash": -20.0, "outside": -20.0, "timeout": -15.0}.get(event, 0.0)
    parts = {"dock": dock, "route": route, "smooth": smooth, "terminal": terminal}
    return float(sum(parts.values())), parts, float(target)


def redistribute(episode, steps=10, decay=0.5):
    if not episode or steps == 0 or episode[-1].get("event") != "touchdown":
        return episode
    if steps < 0 or not 0.0 < decay <= 1.0:
        raise ValueError("Invalid redistribution parameters")
    target = episode[-1]["target_reward"]
    count = min(steps, len(episode))
    weights = decay ** np.arange(count, dtype=float)
    weights /= weights.sum()
    episode[-1]["reward"] -= target
    for i, weight in enumerate(weights):
        episode[-1 - i]["reward"] += float(weight * target)
    return episode
