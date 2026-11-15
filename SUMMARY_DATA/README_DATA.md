# 摘要数据 v1.2

训练池有 14,733 个独立 Reddit TL;DR 问题，导出 500、2,000、10,000 三档嵌套预算。四类 chosen/rejected 来源组合在三档和 A/B 间比例一致。使用对应的 `arms/` 文件训练。用 `test_200_main.jsonl` 作为主评测，`test_500_full.jsonl` 用于扩展评测。生成时读 `messages`，参考摘要另存在同条记录的 `reference`。

`manifests/manifest.json` 记录训练数据和评测资产的 SHA-256。配套讲解见共享目录 `00_docs/SUMMARIZATION_DATA_WALKTHROUGH.md`。训练和测试代码见 `02_code/data_preparation/`。

两套任务模型共用本目录的 MMLU 和 ARC-Challenge，报告 MMLU、ARC 选项字母、ARC 完整选项文本三项准确率。两项 ARC 使用同一组 300 题。LCWR 使用本目录的 TL;DR 测试题。

请把两个 ZIP 解压到同一父目录。IF 的任务主指标使用 IF 包中的 IFEval。摘要清单引用该文件以复查既有数据隔离和复现原测试。
