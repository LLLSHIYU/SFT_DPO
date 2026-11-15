# 02_code · 代码快照

## 文件

| 文件 | 说明 |
|---|---|
| `DSAI4207_code_20261021.zip` | 约 128KB，29 个文件；包含 `data_engineering/`、`README.md`、`requirements.txt`、`.gitignore` |
| `README_CODE.md` | 本文件 |

## 来源

- 打包日期：2026-10-21
- 本地 Git 仓库最后 commit：`e3ac629`（`Complete data/v1.0 official dataset & tokenizer pipeline and v1.2 re-audit fixes`）
- 压缩包是**工作区快照**：包含 commit `e3ac629` 之后的 README / DATA_FIX_PLAN 更新，因此与 commit 不一定逐字节一致。
- 已排除：`__pycache__/`、`*.pyc`、`data/`、`raw_cache/`、`.DS_Store`。

## 使用

```bash
unzip DSAI4207_code_20261021.zip -d DSAI4207_code
cd DSAI4207_code
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **验证前必须安装完整 `requirements.txt`**：`07_validate.py` 会读取 `trl` / `torch` / `transformers` / `accelerate` 版本，测试套件还会跑 1-step TRL SFT/DPO。数据-only 环境会报 `PackageNotFoundError`。

## 目录内容

```text
data_engineering/
  DATA_PROTOCOL.md
  DATA_ISSUES.md
  DATA_FIX_PLAN.md
  configs/data_v1.yaml
  scripts/
    00_download.py
    01_canonicalize.py
    02_decontaminate_and_dedup.py
    03_testset.py
    04_split.py
    05_warmup.py
    06_format_arms.py
    07_validate.py
    ifeval_harness.py
    instruction_following_eval/
    run_all.py
    utils.py
  tests/test_data_invariants.py
README.md
requirements.txt
.gitignore
```

## 注意

- 代码 zip 里**没有**数据；数据在 `../01_data/v1.0/`。
- 不要用 zip 里的 `data_engineering/scripts` 去覆盖共享数据；数据已冻结。
- 需要完整 Git 历史时，用本地仓库或向 PAN Haoren 要 Git bundle；本 zip 只是快照。
