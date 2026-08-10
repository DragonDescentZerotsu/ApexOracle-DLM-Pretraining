# ApexOracle-DLM-Pretraining 维护约束

- 本仓库只维护合作者开发的 DLM + 209-descriptor MTR 预训练 producer；downstream embedding、guidance heads
  和 candidate scoring 属于 `ApexOracle-MDLM`。
- 原始 source-only 状态由 branch `legacy/apexoracle-dlm-pretraining-2026-08-10`、commit
  `fda167cf5fb90ac57952482fb5c0e605b188c105` 与 annotated tag `legacy-code-snapshot-2026-08-10` 恢复。
- 不改变模型结构、DLM objective、MTR loss 权重、tokenizer vocabulary 或 checkpoint key schema；只允许修复
  阻塞 portability、安装和验证的问题，并在 `docs/PORTABILITY_AUDIT.md` 登记。
- Canonical 训练入口为 `python main.py model=medium data=SELFIES ...`。数据路径由
  `APEXORACLE_DLM_TRAIN_PATH`、`APEXORACLE_DLM_VALID_PATH` 和 `APEXORACLE_DLM_CACHE_DIR` 提供；输出由
  Hydra 写入 `outputs/`，checkpoint 写入 `${checkpointing.save_dir}/checkpoints/`，二者均不得进入 Git。
- Canonical synthetic 验证为 `python scripts/smoke_train_save_load.py --device cuda:0`；它执行一次
  DLM-token + MTR-descriptor 联合优化步、保存和 strict reload，不声称重现完整预训练数据或正式 checkpoint。
- 公共提交前运行 `python -m pytest -q`、`python -m compileall -q .`、`git diff --check`，并审计 Git tree
  中不存在 checkpoint、dataset、cache、W&B 或 raw output。
