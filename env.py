from collections import deque
import inspect
import math
import threading
import time
from types import SimpleNamespace

import airsim
import numpy as np
from pythonosc import dispatcher, osc_server, udp_client
from scipy.spatial.transform import Rotation

from config import TaskConfig
from gait import WalkingPose
from task import RelativeKF, Route, State, contact_metrics, infer_action, normalize, predict_state, spherical_velocity, team_reward, wrap


def rpc_compatibility():
    module = airsim.client.msgpackrpc
    signature = inspect.signature(module.Client)
    if "pack_encoding" in signature.parameters or any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values()):
        return

    def decode(value):
        if isinstance(value, dict):
            return {(key.decode("utf-8") if isinstance(key, bytes) else key): decode(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(decode(item) for item in value)
        return value

    class Client(module.Client):
        def __init__(self, *args, **kwargs):
            kwargs.pop("pack_encoding", None)
            kwargs.pop("unpack_encoding", None)
            super().__init__(*args, **kwargs)

        def call(self, *args, **kwargs):
            return decode(super().call(*args, **kwargs))

    airsim.client.msgpackrpc = SimpleNamespace(**{**vars(module), "Client": Client})


class Env:
    observation_dims = (22, 28)
    action_dims = (3, 6)

    def __init__(self, config=None, seed=43342521, manual=False):
        self.config = config or TaskConfig()
        self.seed, self.manual = seed, manual
        self.phase, self.episode = 1, -1
        rpc_compatibility()
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.dog_client = udp_client.SimpleUDPClient(self.config.host, self.config.send_port)
        self.lock = threading.Condition()
        self.status, self.status_seq = None, 0
        self.marker, self.contact = None, None
        disp = dispatcher.Dispatcher()
        disp.map("/status", self._receive_status)
        disp.map("/marker", self._receive_marker)
        disp.map("/contact", self._receive_contact)
        self.server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", self.config.listen_port), disp)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.wind_counter, self.wind = 0, np.zeros(3)
        self.closed = False

    def _receive_status(self, address, *args):
        if len(args) < 9:
            return
        values = np.asarray(args[:9], dtype=float)
        if np.isfinite(values).all():
            with self.lock:
                self.status = values
                self.status_seq += 1
                self.lock.notify_all()

    def _receive_marker(self, address, *args):
        if len(args) == 5 and np.isfinite(args).all():
            with self.lock:
                self.marker = (bool(args[0]), np.asarray(args[1:4]), float(args[4]))

    def _receive_contact(self, address, *args):
        if len(args) == 6 and np.isfinite(args).all():
            with self.lock:
                if self.contact is None:
                    self.contact = {"time": float(args[0]), "landing_gear": bool(args[1]),
                                    "platform_top": bool(args[2]), "point": list(args[3:6])}

    def _snapshot(self, wait=False):
        with self.lock:
            if wait and not self.lock.wait_for(lambda: self.status is not None, self.config.state_timeout):
                raise TimeoutError("No UGV /status packet")
            return (None if self.status is None else self.status.copy()), self.status_seq

    def set_phase(self, phase):
        if phase not in (1, 2, 3):
            raise ValueError("Phase must be 1, 2, or 3")
        if phase != self.phase:
            self.wind_counter = 0
        self.phase = phase

    def _set_wind(self):
        if self.phase != 3:
            self.wind = np.zeros(3)
        elif self.wind_counter == 0:
            angle = self.wind_rng.uniform(-math.pi, math.pi)
            speed = self.config.wind_speed * math.sqrt(self.wind_rng.uniform())
            self.wind = np.array([speed * math.cos(angle), speed * math.sin(angle), 0.0])
        self.client.simSetWind(airsim.Vector3r(*self.wind.tolist()))
        self.wind_counter = (self.wind_counter + 1) % self.config.wind_interval

    def _uav_state(self, message=None):
        message = self.client.getMultirotorState() if message is None else message
        self.uav_timestamp = int(message.timestamp)
        state = message.kinematics_estimated
        flip = np.diag([1.0, 1.0, -1.0])
        q = state.orientation
        rotation = flip @ Rotation.from_quat([q.x_val, q.y_val, q.z_val, q.w_val]).as_matrix() @ flip
        vector = lambda v: np.array([v.x_val, v.y_val, v.z_val], dtype=float)
        position, velocity = flip @ vector(state.position), flip @ vector(state.linear_velocity)
        omega = rotation @ (-flip @ vector(state.angular_velocity))
        return State(position, rotation, velocity, omega, rotation.T @ velocity, 0.0)

    def _ground_state(self, raw, duration=None):
        cfg = self.config
        frame = Rotation.from_euler("z", cfg.ue_heading).as_matrix()
        position = frame @ (raw[:3] / 100.0 - np.asarray(cfg.ue_origin))
        rotation = frame @ Rotation.from_euler("xyz", [-raw[3], -raw[4], raw[5]], degrees=True).as_matrix()
        body_velocity = np.array([raw[6], raw[7], math.radians(raw[8])])
        velocity = rotation @ np.r_[body_velocity[:2], 0.0]
        omega = frame @ np.array([0.0, 0.0, body_velocity[2]])
        if duration is not None and duration > 1e-8:
            velocity = (position - self.measured_body.position) / duration
            omega = Rotation.from_matrix(rotation @ self.measured_body.rotation.T).as_rotvec() / duration
        height = cfg.nominal_height + position[2]
        body = State(position, rotation, velocity, omega, body_velocity, height)
        offset = rotation @ np.asarray(cfg.platform_offset)
        platform = State(position + offset, rotation, velocity + np.cross(omega, offset), omega, body_velocity, height)
        return body, platform

    def _capture(self):
        self.uav = self._uav_state()
        stamp = (self.uav_timestamp - self.epoch) / 1e9
        raw, seq = self._snapshot()
        fresh = raw is not None and seq != self.measured_seq
        if fresh:
            self.body, self.ugv = self._ground_state(raw, stamp - self.measured_time)
            self.measured_body, self.measured_ugv = self.body, self.ugv
            self.measured_time, self.measured_seq = stamp, seq
        else:
            age = stamp - self.measured_time
            if age > self.config.state_timeout:
                raise TimeoutError("UGV feedback stopped advancing")
            self.body = predict_state(self.measured_body, age)
            self.ugv = predict_state(self.measured_ugv, age)
        self.elapsed = stamp
        if fresh:
            self.history.append((stamp, self.ugv.position - self.uav.position, self.ugv.rpy.copy()))
        return fresh

    def _run_until(self, target):
        start = self.uav_timestamp
        duration = (target - start) / 1e9
        if duration <= 0:
            return
        self.client.simContinueForTime(duration)
        deadline = time.monotonic() + self.config.step_timeout
        while True:
            message = self.client.getMultirotorState()
            if self.client.simIsPause() and int(message.timestamp) > start:
                if int(message.timestamp) + self.config.physics_dt * 1e9 < target:
                    raise RuntimeError("Simulation paused before the requested interval elapsed")
                return
            if time.monotonic() >= deadline:
                self.client.simPause(True)
                raise TimeoutError("Simulation clock did not complete the requested interval")
            time.sleep(0.001)

    def _observe(self, initial=False, marker_visible=True, duration=None):
        cfg = self.config
        target_time = self.elapsed - cfg.latency_ms / 1000.0
        candidates = [entry for entry in self.history if entry[0] <= target_time + 1e-8]
        sample = candidates[-1] if candidates else None
        stamp, relative, attitude = sample if sample is not None else (0.0, self.last_relative, self.ugv.rpy)
        relative = relative.copy()
        visible = bool(marker_visible and sample is not None
                       and self.elapsed - stamp <= cfg.dt + cfg.latency_ms / 1000.0 + cfg.physics_dt)
        if cfg.external_marker:
            with self.lock:
                marker = self.marker
            visible = bool(marker_visible and marker is not None and marker[0]
                           and 0 <= self.elapsed - marker[2] <= cfg.dt + cfg.physics_dt)
            if visible:
                _, relative, stamp = marker
                relative = relative.copy()
                visible = stamp <= self.elapsed + 1e-8
        if visible and self.sensor_rng.random() < cfg.packet_loss:
            visible = False
        if visible and self.phase == 3:
            relative += self.sensor_rng.normal(0.0, cfg.sensor_noise, 3)
        if initial:
            self.kf = RelativeKF(relative[:2], self.ugv.velocity[:2] - self.uav.velocity[:2], cfg)
            self.last_relative = relative.copy()
        else:
            corrected = self.kf.update(relative[:2] if visible else None, duration, stamp)
            if corrected:
                self.last_relative = relative.copy()
            relative = self.last_relative.copy()
            relative[:2] = self.kf.x[:2]
        self.marker_losses = 0 if visible else self.marker_losses + 1
        attitude_uav = normalize(self.uav.rpy, math.pi)
        attitude_ugv = normalize(attitude, math.pi)
        uav = np.r_[normalize(relative, cfg.safety_radius), attitude_uav, attitude_ugv,
                    normalize(self.uav.velocity, cfg.uav_speed), normalize(self.kf.predictions().ravel(), cfg.safety_radius)]
        ugv = np.r_[normalize(self.ugv.position - self.uav.position, cfg.safety_radius), attitude_uav,
                    normalize(self.ugv.body_velocity, [cfg.ugv_speed, cfg.ugv_speed, cfg.ugv_yaw_rate]),
                    normalize([self.ugv.height, self.ugv.rpy[1], self.ugv.rpy[0]],
                              [cfg.maximum_height, cfg.posture_limit, cfg.posture_limit]),
                    normalize(self.uav.velocity, cfg.uav_speed), normalize(self.ugv.rpy[2], math.pi),
                    self.route.observation(self.body, self.nominal)]
        return uav.astype(np.float32), ugv.astype(np.float32)

    def reset(self, episode=None):
        cfg = self.config
        self.episode = self.episode + 1 if episode is None else episode
        seeds = np.random.SeedSequence([self.seed, self.episode]).spawn(4)
        rng, gait_rng, self.wind_rng, self.sensor_rng = [np.random.default_rng(seed) for seed in seeds]
        self.gait, self.route = WalkingPose(cfg, gait_rng), Route.generate(rng, cfg)
        self.steps, self.elapsed, self.marker_losses = 0, 0.0, 0
        self.done = False
        self.history = deque(maxlen=2000)
        self.previous_uav_action, self.previous_ugv_action = np.zeros(3), np.zeros(6)
        self.yaw_integral, self.yaw_previous_error = 0.0, 0.0
        self.wind_counter = 0
        self.client.simPause(False)
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.simPause(True)
        with self.lock:
            self.status, self.marker, self.contact = None, None, None
        self.dog_client.send_message("/reset", [1.0])
        spawn = np.array([rng.uniform(-15.0, -5.0), rng.uniform(-5.0, 5.0), rng.uniform(3.0, 8.0)])
        self.client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(spawn[0], spawn[1], -spawn[2]), airsim.to_quaternion(0, 0, 0)), True)
        self._set_wind()
        self.wind_counter = 0
        idle = np.array([0.0, 0.0, 0.0, cfg.nominal_height, 0.0, 0.0])
        self._send_control(idle)
        self.uav = self._uav_state()
        self.client.moveByVelocityAsync(0, 0, 0, cfg.dt + cfg.physics_dt)
        self._run_until(self.uav_timestamp + round(cfg.dt * 1e9))
        raw, seq = self._snapshot(wait=True)
        self.uav = self._uav_state()
        self.epoch = self.uav_timestamp
        self.body, self.ugv = self._ground_state(raw)
        self.measured_body, self.measured_ugv = self.body, self.ugv
        self.measured_time, self.measured_seq = 0.0, seq
        self.nominal = self.route.command(self.body.position, self.body.rpy[2])
        self.last_relative = self.ugv.position - self.uav.position
        self.history.append((0.0, self.last_relative.copy(), self.ugv.rpy.copy()))
        self.metrics = contact_metrics(self.uav, self.ugv, cfg)
        self.collision_stamp = int(self.client.simGetCollisionInfo().time_stamp)
        with self.lock:
            self.marker, self.contact = None, None
        if self.manual:
            self.client.enableApiControl(False)
        return self._observe(initial=True)

    def _ugv_command(self, action):
        return self.nominal + np.asarray(self.config.residual) * action

    def _send_control(self, command):
        cfg = self.config
        physical = command.copy()
        physical[:2] += self.gait.linear_velocity[:2] + command[2] * np.array([-self.gait.position[1], self.gait.position[0]])
        speed = np.linalg.norm(physical[:2])
        if speed > cfg.ugv_speed:
            physical[:2] *= cfg.ugv_speed / speed
        physical[2] = np.clip(command[2] + self.gait.yaw_rate, -cfg.ugv_yaw_rate, cfg.ugv_yaw_rate)
        posture = Rotation.from_euler("xyz", [command[5], command[4], 0]).as_matrix() @ self.gait.rotation
        roll, pitch, _ = Rotation.from_matrix(posture).as_euler("xyz")
        physical[4:] = np.clip([pitch, roll], -cfg.posture_limit, cfg.posture_limit)
        minimum = cfg.minimum_height + 0.75 * abs(math.sin(physical[4])) + 0.5 * abs(math.sin(physical[5]))
        physical[3] = np.clip(command[3] + self.gait.position[2], minimum, cfg.maximum_height)
        self.dog_client.send_message("/control", [float(physical[0]), float(physical[1]), math.degrees(physical[2]),
                                                 float(physical[3] / cfg.maximum_height * 100.0),
                                                 -math.degrees(physical[4]), -math.degrees(physical[5])])
        return physical

    def _yaw_rate(self):
        delta = self.ugv.position - self.uav.position
        target = math.atan2(delta[1], delta[0]) if np.linalg.norm(delta) > 2.0 else self.ugv.rpy[2]
        error, cfg = wrap(target - self.uav.rpy[2]), self.config
        self.yaw_integral = np.clip(self.yaw_integral + error * cfg.dt, -math.pi, math.pi)
        derivative = wrap(error - self.yaw_previous_error) / cfg.dt
        self.yaw_previous_error = error
        return float(np.clip(cfg.yaw_kp * error + cfg.yaw_ki * self.yaw_integral + cfg.yaw_kd * derivative, -cfg.yaw_limit, cfg.yaw_limit))

    def _collision_event(self, collision, contact, uav, ugv):
        cfg = self.config
        if contact is not None:
            target = bool(contact["landing_gear"] and contact["platform_top"])
            return "touchdown" if target else "crash", np.asarray(contact["point"], dtype=float), "scene_contact"
        if collision is None:
            return "running", None, "none"
        impact = np.array([collision.impact_point.x_val, collision.impact_point.y_val, -collision.impact_point.z_val])
        local = ugv.rotation.T @ (impact - ugv.position)
        normal = np.array([collision.normal.x_val, collision.normal.y_val, -collision.normal.z_val])
        matched = not cfg.platform_object or cfg.platform_object.lower() in str(collision.object_name).lower()
        on_surface = (abs(local[0]) <= cfg.platform_half_size[0] and abs(local[1]) <= cfg.platform_half_size[1]
                      and abs(local[2]) <= cfg.contact_tolerance)
        top = float(np.dot(normal, ugv.rotation[:, 2])) >= cfg.contact_normal_cosine
        above = np.dot(uav.position - ugv.position, ugv.rotation[:, 2]) >= -cfg.contact_tolerance
        return "touchdown" if matched and on_surface and top and above else "crash", impact, "collision_geometry"

    def step(self, uav_action=None, ugv_action=None, marker_visible=True, contact=None):
        if self.done:
            raise RuntimeError("Reset the environment before stepping a finished episode")
        cfg = self.config
        if not self.manual:
            uav_action = np.asarray(uav_action, dtype=float)
            if uav_action.shape != (3,) or not np.isfinite(uav_action).all():
                raise ValueError("UAV action must contain three finite values")
            uav_action = np.clip(uav_action, -1.0, 1.0)
        ugv_action = np.zeros(6) if ugv_action is None or self.phase == 1 or self.manual else np.asarray(ugv_action, dtype=float)
        if ugv_action.shape != (6,) or not np.isfinite(ugv_action).all():
            raise ValueError("UGV action must contain six finite values")
        ugv_action = np.clip(ugv_action, -1.0, 1.0)
        previous_uav, previous_ugv, previous_body = self.uav, self.ugv, self.body
        start_elapsed = self.elapsed
        nominal_velocity = Rotation.from_euler("z", self.body.rpy[2]).apply(np.r_[self.nominal[:2], 0.0])
        command = self._ugv_command(ugv_action)
        self._set_wind()
        if not self.manual:
            velocity = spherical_velocity(uav_action, cfg.uav_speed)
            self.client.moveByVelocityAsync(float(velocity[0]), float(velocity[1]), -float(velocity[2]), cfg.dt + cfg.physics_dt,
                                           yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=math.degrees(self._yaw_rate())))
        times, velocities = [0.0], [self.uav.velocity.copy()]
        collision = None
        event_uav, event_ugv = self.uav, self.ugv
        count = int(math.ceil(cfg.dt / self.gait.dt))
        for index in range(count):
            before_uav, before_ugv, before_time = self.uav, self.ugv, self.elapsed
            physical = self._send_control(command)
            target = self.epoch + round((self.steps * cfg.dt + min((index + 1) * self.gait.dt, cfg.dt)) * 1e9)
            self._run_until(target)
            self._capture()
            duration = self.elapsed - before_time
            self.gait.advance(self.ugv.body_velocity, duration)
            sample = self.client.simGetCollisionInfo()
            with self.lock:
                scene_contact = self.contact
            if contact is None and scene_contact is not None and before_time - 1e-8 <= scene_contact["time"] <= self.elapsed + 1e-8:
                contact = scene_contact
            if sample.has_collided and int(sample.time_stamp) > self.collision_stamp:
                collision = sample
                self.collision_stamp = int(sample.time_stamp)
            if contact is not None or collision is not None:
                contact_time = contact.get("time", self.elapsed) if contact is not None else (int(collision.time_stamp) - self.epoch) / 1e9
                delta = float(np.clip(contact_time - before_time, 0.0, duration))
                event_uav, event_ugv = predict_state(before_uav, delta), predict_state(before_ugv, delta)
                break
            times.append(self.elapsed - start_elapsed)
            velocities.append(self.uav.velocity.copy())
        quality = {"valid": True, "rmse": 0.0}
        if self.manual:
            prior = spherical_velocity(self.previous_uav_action, cfg.uav_speed) if self.steps else previous_uav.velocity
            uav_action, quality = infer_action(times, velocities, prior, cfg)
        duration = self.elapsed - start_elapsed
        self.steps += 1
        observation = self._observe(marker_visible=marker_visible, duration=duration)
        event, impact, source = self._collision_event(collision, contact, event_uav, event_ugv)
        self.metrics = contact_metrics(event_uav, event_ugv, cfg, impact) if event != "running" else contact_metrics(self.uav, self.ugv, cfg)
        if event == "running":
            if self.metrics["distance"] > cfg.safety_radius:
                event = "outside"
            elif self.steps >= cfg.episode_steps or self.marker_losses >= cfg.marker_loss_steps:
                event = "timeout"
        self.done = event != "running"
        progress = self.route.progress(previous_body.position, self.body.position)
        nominal_progress = self.route.progress(previous_body.position, previous_body.position + nominal_velocity * duration)
        _, _, lateral, heading = self.route.geometry(self.body.position, self.body.rpy[2])
        previous_distance = self.metrics["distance"] if self.steps == 1 else float(np.linalg.norm(previous_uav.position - previous_ugv.position))
        reward, parts, target_reward = team_reward(previous_distance, self.metrics, progress, nominal_progress, lateral, heading,
                                                  uav_action, ugv_action, self.previous_uav_action, self.previous_ugv_action, event)
        self.previous_uav_action, self.previous_ugv_action = uav_action.copy(), ugv_action.copy()
        self.route.advance(self.body.position)
        self.nominal = self.route.command(self.body.position, self.body.rpy[2])
        observation = (observation[0], np.r_[observation[1][:-12], self.route.observation(self.body, self.nominal)].astype(np.float32))
        info = {"event": event, "touchdown": event == "touchdown", "safe_landing": event == "touchdown" and self.metrics["safe"],
                "target_reward": target_reward, "reward_components": parts, "metrics": self.metrics.copy(),
                "uav_action": uav_action.tolist(), "ugv_action": ugv_action.tolist(), "ugv_command": physical.tolist(),
                "wind": self.wind.tolist(), "phase": self.phase, "route_finished": self.route.finished,
                "contact_source": source, "elapsed": self.elapsed, "expert_valid": quality["valid"]}
        return observation, reward, self.done, info

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            cfg = self.config
            self.dog_client.send_message("/control", [0.0, 0.0, 0.0, cfg.nominal_height / cfg.maximum_height * 100.0, 0.0, 0.0])
            self.client.simSetWind(airsim.Vector3r(0, 0, 0))
            self.client.simPause(False)
            if not self.manual:
                self.client.moveByVelocityAsync(0, 0, 0, cfg.dt).join()
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
        finally:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(timeout=2.0)


