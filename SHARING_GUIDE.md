# OneDrive 共享教程（DSAI 4207）

本目录 `DSAI4207_shared/` 已经建好并填充完毕。下面给出 **macOS + OneDrive** 的共享步骤；如果你用的是 Windows 或 PolyU SharePoint/Teams，见第 5–6 节。

> **2026-10-28 状态：** 本机已安装并登录 OneDrive（PolyU 账号），共享目录已复制到：
>
> ```text
> ~/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/DSAI4207_shared
> ```
>
> 已校验：83 个文件、285MB、`CHECKSUMS.sha256` 0 失败。**下一步只需要在 Finder 右键共享**（见第 3 节）。

---

## 1. 安装并登录 OneDrive（macOS）

### 1.1 用 PolyU 账号（推荐）

1. 打开 PolyU 的 Microsoft 365 入口：`https://www.office.com`（或 PolyU IT 提供的 Microsoft 365 页面）。
2. 用你的 PolyU NetID / `@connect.polyu.hk` 邮箱登录。
3. 在 Microsoft 365 首页点 **OneDrive**，确认能看到你的网盘。
4. 安装 OneDrive for Mac：
   - 在 OneDrive 网页左下角点 **Get the OneDrive apps**；或
   - 从 Mac App Store 搜索 “OneDrive”；或
   - 用 PolyU IT 软件中心提供的安装包。
5. 打开 OneDrive，登录同一个 PolyU 账号。
6. 选择同步目录。默认会是：
   ```text
   ~/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/
   ```
   或旧版客户端：
   ```text
   ~/OneDrive - The Hong Kong Polytechnic University/
   ```
7. 等待首次同步完成（菜单栏 OneDrive 图标显示 “Up to date”）。

### 1.2 用个人 Microsoft 账号

步骤相同，只是登录个人账号，目录通常是 `~/OneDrive`。课程数据建议用 PolyU 账号，权限和合规更清楚。

---

## 2. 把共享目录放进 OneDrive

### 方法 A：Finder 拖拽（推荐）

1. 在 Finder 中打开：
   ```text
   /Users/panhaoren/Documents/学习/PolyU/dsai4207/
   ```
2. 找到 `DSAI4207_shared` 文件夹。
3. 拖到 OneDrive 同步目录（例如 `~/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/`）。
4. 等待 OneDrive 上传：
   - 菜单栏 OneDrive 图标 → “Up to date”；
   - Finder 中文件夹旁出现绿色对勾。
5. 数据约 **285MB、51 个文件**，首次上传取决于网速；上传期间不要关机/休眠。

### 方法 B：命令行复制

```bash
# 把 <OneDrive目录> 替换成你的实际路径
rsync -a --progress \
  "/Users/panhaoren/Documents/学习/PolyU/dsai4207/DSAI4207_shared" \
  "$HOME/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/"
```

> **不要用 symlink/替身**。OneDrive 对 macOS symlink 支持不稳定；本目录已经是真实文件副本，直接复制即可。

---

## 3. 在 OneDrive 中共享

### 3.1 共享整个文件夹

1. 在 Finder 中右键 `DSAI4207_shared` → **Share**（或 **OneDrive → Share**）。
2. 选择 **Specific people**（特定人员），不要选 “Anyone with the link”。
3. 输入三位组员的 PolyU 邮箱：
   - PAN Haoren
   - LIU Shiyu
   - PAN Yubing
4. 权限建议：
   - **Can view**：`01_data/`、`04_results/`、`05_handoff/`
   - **Can edit**：`00_docs/`、`03_runs/`
   - 如果 OneDrive 只允许对整个文件夹设一个权限，先设 **Can view**，再单独把 `00_docs`、`03_runs` 设为 **Can edit**。
5. 可选：
   - 设置链接过期时间（例如 2026-12-31）；
   - 设置密码（如果 OneDrive 界面提供）。
6. 点 **Send** 或 **Copy link**，把链接发到小组群。
7. 让组员打开链接确认能访问；最好让其中一人下载 `01_data/v1.0/DATA_CARD.md` 验证权限。

### 3.2 只共享某个子目录

- 只想给训练组数据：右键 `01_data` → Share → **Can view**。
- 只想给评测组结果：右键 `04_results` → Share → **Can view**。
- 会议记录/文档需要共同编辑：右键 `00_docs` → Share → **Can edit**。

### 3.3 权限检查

- 打开 OneDrive 网页 → 右键文件夹 → **Manage access**；
- 确认：
  - 没有 “Anyone with the link”；
  - 没有校外人员；
  - `01_data` 对组员是 **Can view**；
  - 链接没有意外设置为 “Allow editing”。

---

## 4. 组员如何接收

1. 点开群里的 OneDrive 链接。
2. 选择 **Download** 或 **Add shortcut to My files**（推荐，直接在 OneDrive 里看到）。
3. 不要直接在共享的 `01_data/v1.0` 里编辑文件；需要改数据时找 PAN Haoren 出新版本 `v1.1`。
4. 训练/评测产物按 `03_runs/README_RUNS.md`、`04_results/README_RESULTS.md` 的命名放到对应目录。
5. 如果看到 “Read-only” 提示，这是预期的：`01_data` 是冻结数据。

---

## 5. 不想装客户端：OneDrive 网页上传

1. 打开 `https://www.office.com` → OneDrive。
2. **New → Folder**，命名 `DSAI4207_shared`。
3. 进入文件夹，按六个子目录分别 **Upload**：
   - `00_docs`
   - `01_data`
   - `02_code`
   - `03_runs`
   - `04_results`
   - `05_handoff`
4. 285MB 数据用浏览器上传可能较慢/中断；建议用第 1 节的同步客户端。
5. 上传完成后按第 3 节共享。

---

## 6. 用 PolyU SharePoint / Teams 代替

如果组里已经有 Teams 团队：

1. 打开 Teams → 对应 Team → **Files**；
2. **Upload** 整个 `DSAI4207_shared`；
3. 右键文件夹 → **Share** → 添加三位组员；
4. 权限同样建议：数据只读、文档可编辑；
5. 优点：PolyU 账号权限统一，适合课程项目；缺点：如果 Team 是公开的，注意不要把数据放到公开频道。

---

## 7. 常见问题

| 问题 | 处理 |
|---|---|
| 上传很慢/中断 | 保持 Mac 唤醒；用有线网络；分两次上传（先 `00_docs`+`02_code`，再 `01_data`） |
| 出现 “ conflicted copy” | 两个人同时改了同一文件；保留正确版本，删除冲突副本 |
| 组员说只能看不能改 | 正常：`01_data` 设为 Can view；需要编辑的是 `00_docs` / `03_runs` |
| 链接被校外看到 | 立即 **Manage access** 移除；改为 Specific people |
| 文件名含中文/空格 | OneDrive 支持；避免 `:`、`?`、`*` 等 Windows 非法字符 |
| 数据文件是只读 | 预期；需要更新时由数据负责人新建 `data/v1.1`，不要改 `v1.0` |
| 想回滚文档 | OneDrive 右键 → **Version history**；不要靠 `v1/v2/v3` 文件名堆版本 |

---

## 8. 合规提醒

- `01_data` 来自 Reddit TL;DR，含真实用户发帖，可能含 PII/情绪化内容；**只共享给课程小组**，不要公开链接、不要用于商业用途。
- 共享链接优先用 **Specific people**，不要用 “Anyone with the link”。
- 课程结束后按学校/课程要求删除或归档共享链接。

---

## 9. 当前状态与我还能帮你做什么

### 已完成（2026-10-28）

- OneDrive.app 已安装并登录 PolyU 账号。
- `DSAI4207_shared` 已复制到：
  ```text
  ~/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/DSAI4207_shared
  ```
- 复制后校验：83 个文件、285MB、`CHECKSUMS.sha256` 0 失败。
- 已在 Finder 中打开 OneDrive 副本。

### 还没完成（需要你在 Finder/OneDrive 里点）

1. 右键 `DSAI4207_shared` → **Share**；
2. 添加三位组员的 PolyU 邮箱；
3. 设置权限（数据/结果只读，文档/run 可编辑）；
4. **Copy link** 发群。

你完成共享后，把链接或权限页面截图发我，我可以帮你检查：
- 是否误设为 “Anyone with the link”；
- 组员是否只有该有的权限；
- 链接是否能打开、文件是否完整。

如果右键没有 **Share** 菜单，先去系统设置 → 隐私与安全性 → 扩展 → Finder 扩展，启用 OneDrive Finder Integration。
