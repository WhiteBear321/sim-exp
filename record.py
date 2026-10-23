import argparse
from dataclasses import asdict
import json
from pathlib import Path

from config import ROOT, TaskConfig, TrainConfig
from masac import Replay, SCHEMA, transition
from task import redistribute


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--attempts", type=int, default=500)
    parser.add_argument("--seed", type=int, default=43342521)
    parser.add_argument("--output", type=Path, default=ROOT / "expert_data")
    parser.add_argument("--platform-object", default="")
    parser.add_argument("--response", type=float, nargs=3, default=(0.5, 0.5, 0.5))
    parser.add_argument("--ue-origin", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--ue-heading", type=float, default=0.0)
    parser.add_argument("--external-marker", action="store_true")
    args = parser.parse_args()
    if args.episodes <= 0 or args.attempts < args.episodes:
        raise ValueError("Attempts must be at least the positive requested episode count")
    args.output.mkdir(parents=True, exist_ok=True)
    task = TaskConfig(platform_object=args.platform_object, velocity_response=tuple(args.response),
                      ue_origin=tuple(args.ue_origin), ue_heading=args.ue_heading, external_marker=args.external_marker)
    if any(not 0.0 < value < 1.0 for value in task.velocity_response):
        raise ValueError("Response coefficients must lie strictly between zero and one")
    training = TrainConfig()
    from env import Env
    env = Env(task, args.seed, manual=True)
    saved = 0
    existing = [int(path.stem.split("_")[-1]) for path in args.output.glob("episode_*.json")]
    number = max(existing, default=0)
    try:
        for episode in range(args.attempts):
            observation = env.reset(episode)
            print(f"Attempt {episode + 1}: pilot the UAV with the existing RC controller", flush=True)
            points = env.route.points[1:].tolist()
            speeds = env.route.speeds.tolist()
            initial_position = env.uav.position.tolist()
            pending = []
            done = False
            valid = True
            while not done:
                next_observation, reward, done, info = env.step()
                valid = valid and info["expert_valid"]
                pending.append(transition(observation, next_observation, info, reward, done))
                observation = next_observation
            if info["safe_landing"] and valid:
                redistribute(pending, training.redistribution_steps, training.redistribution_decay)
                for item in pending:
                    Replay.validate(item)
                number += 1
                data = {"metadata": {"schema": SCHEMA,
                                     "reward_redistributed": True, "redistribution_steps": training.redistribution_steps,
                                     "redistribution_decay": training.redistribution_decay, "seed": args.seed, "episode": episode,
                                     "initial_position": initial_position, "waypoints": points, "desired_speeds": speeds,
                                     "contact_source": info["contact_source"], "task_config": asdict(task)}, "trajectory": pending}
                path = args.output / f"episode_{number:05d}.json"
                with path.open("x", encoding="utf-8") as stream:
                    json.dump(data, stream, separators=(",", ":"))
                saved += 1
                print(f"Saved {saved}/{args.episodes}: {path.name}", flush=True)
                if saved >= args.episodes:
                    break
            else:
                print(f"Discarded: {info['event']} | safe={info['safe_landing']} | valid={valid}", flush=True)
    finally:
        env.close()
    if saved < args.episodes:
        raise RuntimeError(f"Saved {saved}/{args.episodes} safe demonstrations in {args.attempts} attempts")


if __name__ == "__main__":
    main()
