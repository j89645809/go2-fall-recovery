# go2-fall-recovery

宇树 Go2 从静态倒地状态起身的摔倒恢复（FRNet 对齐策略，基于 mjlab / MuJoCo）。

项目代码、验收标准、评测和训练命令、附带的模型和结果报告详见 [PUBLIC_RELEASE.md](PUBLIC_RELEASE.md)（英文）。

| 路径 | 内容 |
|---|---|
| `src/tasks/velocity/config/go2_recovery_aligned/` | 恢复任务 `Unitree-Go2-Recovery-FRNet-Aligned` |
| `src/tasks/velocity/config/go2_recovery/` | 早期的恢复任务版本 |
| `src/tasks/velocity/config/go2/` | 恢复任务继承的 Go2 基础环境 |
| `frnet_aligned_assets/` | 恢复任务使用的 Go2 MJCF 模型和网格 |
| `scripts/` | 训练、评测、验收和审计脚本 |
| `checkpoints/`、`data/` | 基准策略和复现训练所需的数据 |
| `public_results/` | 报告中各项评测的汇总结果 |
| `deploy/robots/go2/`、`simulate/` | Go2 控制程序和 unitree_mujoco 仿真器，用于 sim2sim 检查 |

## 安装

参考 [doc/setup_zh.md](doc/setup_zh.md)。之后所有命令都在仓库根目录运行：

```bash
export PYTHONPATH=. MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0
```

## 快速开始

评测基准策略（12 秒时长，与报告数字一致）：

```bash
python scripts/paired_static_acceptance.py \
  --task Unitree-Go2-Recovery-FRNet-Aligned \
  --checkpoint checkpoints/model_14475.pt \
  --device cuda:0 --seed 2345 --horizon 600 \
  --success-profile tiered \
  --output results/tiered_2345.json
```

训练以及残差、保留微调的命令见 [PUBLIC_RELEASE.md](PUBLIC_RELEASE.md)。

## sim2sim 检查（可选）

编译仿真器和 Go2 控制程序（需要先安装 [cyclonedds](https://github.com/eclipse-cyclonedds/cyclonedds.git) 和 [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2.git)）：

```bash
cd simulate && mkdir build && cd build && cmake .. && make -j8
cd ../../deploy/robots/go2 && mkdir build && cd build && cmake .. && make
```

先运行 `./simulate/build/unitree_mujoco`，再在 `deploy/robots/go2/build` 下运行 `./go2_ctrl --network=lo`。

## 许可与致谢

本仓库基于 [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)（Apache-2.0，见 [LICENCE](LICENCE)）修改而来，已删除其他机器人和动作模仿相关的部分。第三方许可证在 `doc/license/`。

依赖 [mjlab](https://github.com/mujocolab/mjlab.git)、[rsl_rl](https://github.com/leggedrobotics/rsl_rl.git)、[mujoco_warp](https://github.com/google-deepmind/mujoco_warp.git) 和 [mujoco](https://github.com/google-deepmind/mujoco.git)。
