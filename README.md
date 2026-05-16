# franka-il-rl

End-to-end imitation learning to RL pipeline for robotic pick-and-place: Behavioral Cloning → DAgger → SAC fine-tuning on Franka Panda in MuJoCo, with ROS2 deployment and TensorRT inference.

## Overview

This project explores a complete imitation-to-reinforcement learning pipeline on a simulated Franka Panda arm. A scripted expert generates demonstrations in MuJoCo, which are first cloned via Behavioral Cloning, then refined interactively via DAgger, and finally fine-tuned with Soft Actor-Critic to surpass the expert. The final policy is exported to ONNX/TensorRT and deployed through a ROS2 Humble inference node, with the entire stack containerized via Docker.

The project emphasizes a side-by-side comparison of three learning paradigms (offline imitation, interactive imitation, online RL) under a single environment and evaluation harness, with ablation studies on demonstration count, network capacity, and warm-start strategies.

## Tech Stack

- **Simulation:** MuJoCo 3.x, Gymnasium
- **Robot:** Franka Panda (mujoco_menagerie)
- **Learning:** PyTorch, custom BC/DAgger/SAC implementations
- **Deployment:** ROS2 Humble, ONNX Runtime, TensorRT (FP16)
- **Infrastructure:** Docker, docker-compose, Weights & Biases

## Project Structure

```
franka-il-rl/
├── configs/          # YAML configs per algorithm
├── envs/             # Gymnasium-compatible MuJoCo environment
├── experts/          # Scripted expert policy for demonstrations
├── algos/            # BC, DAgger, SAC implementations
├── networks/         # MLP and Gaussian policy networks
├── data_utils/       # Replay buffer and demonstration dataset
├── utils/            # Logging and seeding helpers
├── scripts/          # Entry points (training, evaluation, export)
├── data/             # Demonstrations and checkpoints (gitignored)
├── ros2_ws/          # ROS2 workspace for inference deployment
└── docker/           # Training and inference container definitions
```

## Roadmap

### Phase 1 — Foundations & Infrastructure

- [x] **Week 1** — Environment setup, MuJoCo sanity check, RL theory grounding
- [x] **Week 2** — Custom Franka env attempt (archived; see Project Notes)
- [x] **Week 3** — Fetch wrapper, scripted expert, demonstration collection
  - [x] gymnasium-robotics setup, `FetchPickPlaceWrapper`
  - [x] Scripted expert (100% success rate)
  - [x] Demonstration collection pipeline (1000 episodes, HDF5)
  - [x] Train/val/test split (80/10/10)
- [x] **Week 4** — Data pipeline, evaluation harness, baseline metrics, W&B integration

### Phase 2 — Algorithms

- [x] **Week 5** — Behavioral Cloning implementation and training
- [ ] **Week 6** — BC ablation studies (demo count, network capacity, loss function)
- [ ] **Week 7** — DAgger implementation with β-scheduling
- [ ] **Week 8** — DAgger variants exploration, BC vs DAgger sample efficiency study
- [ ] **Week 9** — SAC implementation from scratch, baseline training without warm-start
- [ ] **Week 10** — BC-warmstart SAC fine-tuning, demonstrations in replay buffer

### Phase 3 — Deployment & Evaluation

- [ ] **Week 11** — ROS2 inference node, MuJoCo-ROS2 bridge
- [ ] **Week 12** — ONNX export, TensorRT FP16 engine, latency benchmarking
- [ ] **Week 13** — Docker training & inference containers, docker-compose orchestration
- [ ] **Week 14** — Final ablation studies, technical report, README finalization

## Hardware

Developed and tested on:
- GPU: NVIDIA RTX 3050 Ti (4 GB VRAM)
- CPU: Intel i7 (11th gen)
- RAM: 16 GB
- OS: Ubuntu 22.04, ROS2 Humble

## Getting Started

*Setup instructions and reproduction steps will be documented progressively as each milestone completes.*

## Results

*Experiment results, ablation tables, and learning curves will appear here as Phase 2 progresses.*

## Project Notes

**Week 3 pivot (May 2026)**: Initial attempt used a custom MuJoCo environment
with Franka Panda and mink IK. After significant debugging of PD controller
instability and IK convergence issues, switched to the standard
`gymnasium-robotics` FetchPickAndPlace environment. This trades robot
specificity for a battle-tested baseline, allowing focus on the core IL/RL
algorithms (BC, DAgger, SAC) and the deployment pipeline. The legacy
Franka code is preserved under `*_legacy.py` suffixes for reference.

**Random baseline note**: Random policy yields ~15% success rate on
FetchPickAndPlace, of which ~8% stems from initial states where the
object spawns within the 5cm success threshold of the goal (no policy
action required). This is documented for transparency; the effective
"learning floor" for comparison is ~7%, while BC/DAgger/SAC should
reach 80%+ to demonstrate clear value.

**Week 5 (BC baseline)**: BC trained on 800 trajectories reached 100%
success rate at epoch 30, plateauing through epoch 100. Mean episode
return improved from -42 (early epochs, random-like) to -29 (faster
than expert's -32). This unusually strong BC baseline is likely due to
the short-horizon, state-based, low-DoF nature of FetchPickAndPlace;
typical manipulation BC papers report 60-85%. Ablations in Week 6 will
test robustness across seeds, demo counts, and network capacities.

## License

MIT

## Author

Emin Çağan Apaydın — [github.com/e-cagan](https://github.com/e-cagan)