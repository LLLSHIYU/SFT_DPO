# 指令遵循数据 v1.0

源数据为 Ai2 发布的 [Tülu 3 指令遵循偏好集](https://huggingface.co/datasets/allenai/tulu-3-pref-personas-instruction-following)，固定版本 `cdf475940025c0434a22dbd0bbed336a746942fe`，来源卡标注 ODC-BY。清洗后 19,188 个独立问题，导出 500、2,000、10,000 三档嵌套预算，使用对应的 `arms/` 文件训练。

`eval_benchmarks/ifeval_541.jsonl` 是独立的 IFEval 评测。测试题使用原有 Google IFEval 来源，模型生成后由规则检查器评分。训练记录沿用 Tülu 已有偏好标注。

`manifests/manifest.json` 包含清洗数量、来源、抽样统计与 SHA-256。配套讲解见共享目录 `00_docs/INSTRUCTION_FOLLOWING_DATA_WALKTHROUGH.md`。训练和测试代码见 `02_code/data_preparation/`。
