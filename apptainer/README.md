# LIBERO Apptainer (.sif) — 集群部署

把 LIBERO 仿真环境打成单文件 `.sif`（squashfs = 1 个 inode）。源码（含 bddl_files/init_files/assets）烤进镜像，集群上不需要 bind 源码；演示数据集按需 bind 到 `/workspace/libero/datasets`。

## 构建
```bash
bash apptainer/build.sh     # docker build vla/libero:latest → libero.sif（~4.6GB）
```

## 运行（解耦评测的 sim client）
```bash
# 1) droid.sif 起策略 server（用 LIBERO ckpt）：见 droid_policy_learning/apptainer/run_policy_server.sh
# 2) libero sim client：
TASK_SUITE_NAME=libero_10 NUM_TRIALS_PER_TASK=1 bash apptainer/run_eval.sh
```

## 关键 flag（与 robocasa 同源，详见 ../../robocasa/apptainer/README.md）
- `--writable-tmpfs`：只读 sif，entrypoint 要写 `/workspace/.libero/config.yaml`、run_eval 要写 eval_logs。
- `MUJOCO_GL=osmesa`：CPU 渲染，避开 `--nv` 宿主 GL 库 vs 容器 glibc 冲突；client 无需 GPU。
- `--log_dir /tmp/...`：日志写可写目录。
- server(droid.sif) 端：`--nv` + `LD_LIBRARY_PATH` 容器库优先 + `--writable-tmpfs` + `HF_HOME`（见 run_policy_server.sh）。

## 本机实测
libero.sif `--nv` smoke：torch cuda True、`import libero` OK、`-B /root` 软链修复有效；
跨容器评测客户端连 `localhost:8765` 正常往返（docker 侧已验证 libero_10 全 10 任务跑完）。
