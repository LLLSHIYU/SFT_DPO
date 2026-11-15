# 更新与新增文件指南

**从 2026-10-28 起，以 OneDrive 副本为主副本：**

```text
~/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/DSAI4207_shared
```

本地 `~/Documents/学习/PolyU/dsai4207/DSAI4207_shared` 只作为一次性快照/备份，**不要两边同时改**。

---

## 0. 先记住四条规则

1. **OneDrive 文件夹就是共享主副本**：在同步目录里保存/新建文件，OneDrive 会自动上传，不需要手动“上传”。
2. **数据只读**：`01_data/v1.0` 不要改；要改就新建 `v1.1`。
3. **文档只留最新版**：proposal、protocol、DATA_CARD、HANDOFF 直接覆盖更新；旧版本用 OneDrive 版本历史，不堆 `v1/v2/v3` 文件。
4. **每次改完等 “Up to date”**：菜单栏 OneDrive 图标显示绿色对勾/Up to date 后，再在群里说“已更新”。

---

## 1. 不同文件怎么更新

| 你要改的东西 | 在哪里改 | 怎么同步 |
|---|---|---|
| proposal / 协议 / DATA_CARD / 会议记录 | OneDrive `00_docs/` | 直接保存，OneDrive 自动同步 |
| 数据（新版本） | 项目里生成 `data/v1.1`，再复制到 OneDrive `01_data/v1.1` | 见 §3 |
| 代码 | Git 仓库提交后，导出新的 zip 到 `02_code/` | 见 §4 |
| 训练 checkpoint/log | OneDrive `03_runs/{run}/` | 直接写入，自动同步 |
| 评测表格/图 | OneDrive `04_results/` | 直接写入，自动同步 |
| manifest / 校验和 / HANDOFF | OneDrive `05_handoff/` | 数据或代码变更后更新，见 §5 |

---

## 2. 改文档（最常用）

### 2.1 直接编辑（推荐）

1. 在 Finder 打开：
   ```text
   ~/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/DSAI4207_shared/00_docs
   ```
2. 用 Typora/VS Code/Word 打开要改的文件，保存。
3. 等 OneDrive “Up to date”。
4. 在群里说：改了哪个文件、版本号、一句话变更。

### 2.2 如果先在本地项目里改了

只复制**那一个文件**，不要整目录覆盖：

```bash
PROJ="/Users/panhaoren/Documents/学习/PolyU/dsai4207/group project"
SHARED="$HOME/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/DSAI4207_shared"

cp "$PROJ/data_engineering/DATA_PROTOCOL.md" "$SHARED/00_docs/"
cp "$PROJ/data/v1.0/DATA_CARD.md" "$SHARED/00_docs/"
```

### 2.3 会议记录

在 `00_docs/meeting_notes/` 新建：

```text
2026-10-30_weekly.md
```

不要覆盖上一次的会议记录；会议记录是**按日期累积**，不属于“只留最新版”的文件。

---

## 3. 数据更新（只由数据负责人做）

**原则：`01_data/v1.0` 永久冻结；新数据放 `v1.1`。**

### 3.1 在项目里生成并验证新版本

```bash
cd "/Users/panhaoren/Documents/学习/PolyU/dsai4207/group project"
python data_engineering/scripts/run_all.py
python data_engineering/scripts/07_validate.py
```

### 3.2 复制到 OneDrive（不复制 raw_cache）

```bash
PROJ="/Users/panhaoren/Documents/学习/PolyU/dsai4207/group project"
SHARED="$HOME/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/DSAI4207_shared"

rsync -a --exclude 'raw_cache/' --exclude '.DS_Store' \
  "$PROJ/data/v1.1/" "$SHARED/01_data/v1.1/"

# 设为只读
find "$SHARED/01_data/v1.1" -type f -exec chmod a-w {} \;
```

### 3.3 更新交接文件

```bash
cp "$PROJ/data/v1.1/DATA_CARD.md" "$SHARED/00_docs/DATA_CARD.md"
cp "$PROJ/data/v1.1/manifests/manifest.json" "$SHARED/05_handoff/manifest.json"
cp "$PROJ/data/v1.1/manifests/split_report.json" "$SHARED/05_handoff/split_report.json"
cp "$PROJ/data/v1.1/manifests/decontamination_report.json" "$SHARED/05_handoff/decontamination_report.json"
cp "$PROJ/data/v1.1/raw_cache/sources_lock.json" "$SHARED/05_handoff/sources_lock.json"
```

然后在 `05_handoff/HANDOFF.md` 写清楚：为什么升 v1.1、影响了哪些臂、是否要重跑。

---

## 4. 代码更新

### 4.1 先提交 Git

```bash
cd "/Users/panhaoren/Documents/学习/PolyU/dsai4207/group project"
git status
git add <changed files>
git commit -m "…"
```

> 不要提交 `data/v1.0` 的大 JSONL/parquet；`.gitignore` 已排除，但 `git add` 时仍要检查 `git status`。

### 4.2 导出新 zip 到 OneDrive

```bash
PROJ="/Users/panhaoren/Documents/学习/PolyU/dsai4207/group project"
SHARED="$HOME/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/DSAI4207_shared"
DATE=$(date +%Y%m%d)

cd "$PROJ"
zip -rq "$SHARED/02_code/DSAI4207_code_${DATE}.zip" \
  data_engineering README.md requirements.txt .gitignore \
  -x "*/__pycache__/*" "*.pyc" "*.DS_Store"
```

- 更新 `02_code/README_CODE.md`：写清新 zip 的日期、commit hash、变更。
- 只保留**最新 zip + 一个上一版**；确认新 zip 能用后删除更旧的，避免共享目录膨胀。
- 不要用 zip 覆盖共享数据；zip 只放代码/文档。

---

## 5. 更新 manifest / 校验和

只有**数据、代码或 results** 变了才需要重算；只改文档不用。

```bash
SHARED="$HOME/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/DSAI4207_shared"
cd "$SHARED"

find 01_data 02_code 04_results 00_docs/proposal -type f -print0 \
  | sort -z | xargs -0 shasum -a 256 > /tmp/CHECKSUMS.sha256
mv /tmp/CHECKSUMS.sha256 05_handoff/CHECKSUMS.sha256

# 验证
shasum -a 256 -c 05_handoff/CHECKSUMS.sha256
```

如果 `manifest.json` 也变了，记得同步更新 `05_handoff/manifest.json` 和 `00_docs/DATA_CARD.md`。

---

## 6. 训练/评测产物怎么放

### 训练组

```text
03_runs/{arm}_{budget}_{seed}_{YYYYMMDD}/
  config.yaml
  train.log
  checkpoint/
  README.md
```

- 每个 run 一个目录；不要覆盖别人的 run。
- 重跑加 `_rerun1` 或新日期。
- 大 checkpoint 如果 OneDrive 空间不够，只保留最终 adapter，并和组里确认。

### 评测组

```text
04_results/
  lcwr_{data_version}_{date}.csv
  capability_tax_{data_version}_{date}.csv
  ifeval_{data_version}_{date}.csv
  fig_gain_tax_{data_version}.pdf
  fig_oa_curve_{data_version}.pdf
```

- 原始生成文本放 `03_runs/{run}/eval/`，不要把大文件塞进 `04_results`。
- 每个结果在 `04_results/README_RESULTS.md` 写：脚本、checkpoint、数据版本、日期。

---

## 7. 避免 OneDrive 冲突

1. **不要两个人同时改同一个文件**；约定 owner。
2. 看到 “conflicted copy”：
   - 打开两份，保留正确内容，删除冲突副本；
   - 在群里说一声，避免别人继续改旧版。
3. 大文件上传中不要重命名/移动/删除；等 “Up to date”。
4. 不要用 macOS symlink/替身；OneDrive 同步不稳定。
5. 不要用 `: ? * < > |` 等 Windows 非法字符；中文/空格可以。
6. 磁盘紧张时用 “Free up space” 把不常用大文件转为 online-only，但不要对正在训练使用的数据这么做。

---

## 8. 每次更新后的 30 秒检查

```bash
SHARED="$HOME/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/DSAI4207_shared"
cd "$SHARED"

# 1. 顶层结构没乱
ls

# 2. 数据只读
find 01_data/v1.0 -type f -perm -u+w ! -name README_DATA.md

# 3. 校验和
shasum -a 256 -c 05_handoff/CHECKSUMS.sha256 | grep -c FAILED

# 4. 文件数/大小
find . -type f | wc -l
du -sh .
```

期望：第 2 条无输出；第 3 条输出 `0`；第 4 条文件数与本轮变更后一致。

---

## 9. 最简口诀

- **文档**：在 OneDrive 里直接改，等绿勾。
- **数据**：只加新版本 `v1.1`，不改 `v1.0`；改完更新 manifest/checksums。
- **代码**：先 Git commit，再导出带日期的 zip。
- **run/结果**：按命名直接放 `03_runs` / `04_results`。
- **冲突**：一人一文件，看到 conflicted copy 先停手。
