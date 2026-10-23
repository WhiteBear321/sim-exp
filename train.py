import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from config import ROOT, TaskConfig, TrainConfig
from masac import MASAC, Replay, SCHEMA, transition
from task import redistribute


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=43342521)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--expert-dir", type=Path, default=ROOT / "expert_data")
    parser.add_argument("--no-expert", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--platform-object", default=None)
    parser.add_argument("--sensor-noise", type=float, default=None)
    parser.add_argument("--ue-origin", type=float, nargs=3)
    parser.add_argument("--ue-heading", type=float)
    parser.add_argument("--external-marker", action="store_true", default=None)
    args = parser.parse_args()
    data = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    if data is not None and data.get("schema") != SCHEMA:
        raise ValueError("Incompatible checkpoint schema")
    cfg = TrainConfig(**data["config"]) if data else TrainConfig(seed=args.seed)
    task = TaskConfig(**data["task_config"]) if data else TaskConfig()
    if args.steps is not None:
        cfg.steps = args.steps
    if args.platform_object is not None:
        task.platform_object = args.platform_object
    if args.sensor_noise is not None:
        task.sensor_noise = args.sensor_noise
    if args.ue_origin is not None:
        task.ue_origin = tuple(args.ue_origin)
    if args.ue_heading is not None:
        task.ue_heading = args.ue_heading
    if args.external_marker is not None:
        task.external_marker = args.external_marker
    if cfg.steps <= 0 or task.sensor_noise < 0:
        raise ValueError("Training steps must be positive and sensor noise must be nonnegative")
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    agent = MASAC(cfg, args.device)
    replay = Replay(cfg.replay_size, cfg.seed)
    step, episode = (data["step"], data["episode"]) if data else (0, 0)
    if data:
        agent.restore(data, replay)
    elif not args.no_expert:
        count = replay.load_experts(args.expert_dir)
        if count == 0:
            raise RuntimeError("No demonstrations found. Run record.py first or use --no-expert")
        print(f"Loaded {count} demonstration transitions")
    if step >= cfg.steps:
        print(f"Checkpoint already contains {step} environment steps")
        return
    run = args.resume.parent if args.resume else ROOT / "models" / time.strftime("%Y%m%d_%H%M%S")
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps({"training": asdict(cfg), "task": asdict(task)}, indent=2), encoding="utf-8")
    from env import Env
    env = Env(task, cfg.seed)
    next_save = (step // cfg.save_interval + 1) * cfg.save_interval
    last_phase = None
    stats = {}
    try:
        while step < cfg.steps:
            env.set_phase(cfg.phase(step))
            observation = env.reset(episode)
            pending = []
            episode_return = 0.0
            done = False
            while not done and step < cfg.steps:
                phase = cfg.phase(step)
                env.set_phase(phase)
                if phase != last_phase:
                    print(f"Step {step}: phase {phase}", flush=True)
                    last_phase = phase
                actions = agent.select_action(observation, phase)
                next_observation, reward, done, info = env.step(*actions)
                pending.append(transition(observation, next_observation, info, reward, done))
                stats = agent.update(replay, step, phase, guidance=not args.no_expert)
                observation = next_observation
                episode_return += reward
                step += 1
            replay.extend(redistribute(pending, cfg.redistribution_steps, cfg.redistribution_decay))
            episode += 1
            summary = {"step": step, "episode": episode, "phase": phase, "return": episode_return,
                       "event": info["event"], "safe_landing": info["safe_landing"],
                       "metrics": info["metrics"], "training": stats}
            with (run / "episodes.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(summary) + "\n")
            print(f"Episode {episode} | step {step} | {info['event']} | safe={info['safe_landing']} | return={episode_return:.2f}", flush=True)
            if step >= next_save or step >= cfg.steps:
                agent.save(run / f"step_{step}.pt", step, episode, replay, task)
                next_save = (step // cfg.save_interval + 1) * cfg.save_interval
    finally:
        env.close()


if __name__ == "__main__":
    main()
