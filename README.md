# ACR-MASAC

UAV landing on a moving quadruped with two local actors, shared twin critics, a shared temperature, relative-motion KF predictions, and bounded UGV residual control.

Use the original AirSim 1.8.1 / UE 4.27 / PX4 scene and Python environment. Dependency versions are retained. Set scene physics to 0.003 s. Policy intervals use the simulation clock at 0.1 s; walking updates use 0.01 s targets.

```powershell
python record.py --episodes 100
python train.py --device cuda
python train.py --resume models/RUN/step_10000.pt
```

Run inside this folder after starting UE. Recording preserves native RC control. Each action is fitted from velocity samples over its own interval using a first-order velocity response, then converted to the policy's spherical action. `--response BX BY BZ` sets the per-0.1 s response coefficients, initially 0.5. These coefficients require scene calibration. Only safe episodes with valid fits are saved. Action fields use the normal replay format. `models/` and `expert_data/` start empty; earlier data and weights are incompatible.

`masac.py`: learning/replay; `env.py`: AirSim/OSC; `task.py`: routes/KF/rewards/contact; `gait.py`: walking; `config.py`: settings. Use `--no-expert` for training without demonstrations.

Training switches at steps 25,000 and 50,000: UAV only, UGV only with frozen UAV, then joint training to 150,000 steps. BC and scheduled expert sampling apply only in Phase 1. Later phases sample demonstrations and online transitions uniformly together. Wind and sensor noise apply in Phase 3. Only target-contact rewards are redistributed.

Routes contain 2-4 destination waypoints. Segment ranges are 6-7 / 4-5 / 3-3.75 m respectively, with total length 12-15 m, turns within 60 degrees, and desired speeds 0.6-1.0 m/s. Two-waypoint routes reach at most 14 m.

OSC ports remain 8000/9000. `/control`: `[vx,vy,yaw_rate_deg,height_percent,pitch_deg,roll_deg]`; `/status`: `[x_cm,y_cm,z_cm,roll_deg,pitch_deg,yaw_deg,vx,vy,yaw_rate_deg]`; `/reset`: `[1.0]`. Walking motion goes through these control channels; observations use UE feedback once. Limits apply after combining commands. The supplied walk and idle clips are retained; their forward condition maximum is 1 m/s.

Set `--ue-origin X Y Z` in meters and `--ue-heading RADIANS` to align UE feedback with the AirSim positive-up frame. Configure `platform_offset` and `gear_offset` for the actual scene. The original nine-field status has no capture timestamp; it is associated with the current simulator sample. Reused status snapshots do not trigger KF corrections. KF uses `[x,y,vx,vy]`, acceleration standard deviation 1 m/s2, and measurement standard deviation 0.1 m.

Optional scene inputs use the same OSC listener: `/marker [visible,relative_x,relative_y,relative_z,time]` with `--external-marker`; `/contact [time,landing_gear,platform_top,x,y,z]`. Times are seconds since episode initialization; vectors use the aligned positive-up frame. No ArUco detector or UE component adapter is supplied. Without scene contact labels, AirSim collision geometry provides an approximation; `--platform-object NAME` restricts matching. Contact speeds use the last sample before impact, extrapolated to contact time. Scene integration and training performance require simulator runs.
