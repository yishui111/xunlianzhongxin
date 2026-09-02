
## 🚀 换电脑部署（保证可用）

> **方式 A（推荐 · 100% 保证）**：用 U 盘 / 网盘把「原项目整份文件夹」（含全部大件）复制到新电脑 → 双击 `start.bat` 即可。
>
> **方式 B（代码装配）**：`git clone` 本仓库 → 双击 `assemble.bat` 预检大件 → 按提示补齐缺失项（下载地址见下文/README）→ 双击 `start.bat`。

> 说明：引擎、模型、镜像、运行时等大件体积超过 GitHub 单文件 100MB 上限，**不随仓库分发**；本仓库承载全部自研代码与装配指引，"方式 A"是换机部署最稳路径，"方式 B"适合需要重新下载大件的场景。
# AI 声音训练中心 · 新电脑部署方案（DEPLOY）

> 目标：把本仓库部署到一台新电脑，双击启动即可训练 RVC 换声模型 / GPT-SoVITS 文字驱动模型。
>
> **先说清楚**：本仓库只包含自研的 `train_service`（FastAPI 训练服务）、页面模板、测试与文档。
> 第三方引擎（RVC / GPT-SoVITS）、便携 Python 运行时、ffmpeg、ASR 模型、基座权重**体积巨大，不随仓库分发**，
> 因此部署有两类方式，按你的情况选一种。

---

## 0. 环境要求

| 项目 | 要求 |
| ---- | ---- |
| 操作系统 | Windows 10/11（bat 一键启动）；macOS/Linux 可手动跑 `python train_service/train_api.py`（需自行装依赖） |
| 显卡 | NVIDIA GPU（训练必须 CUDA GPU），显存建议 ≥ 8GB；训练参数按显存自动适配（8G 卡也能训） |
| 磁盘 | 数据集 + 训练产物预留 20GB 以上（中间产物会自动清理） |
| 端口 | 8050（默认，可用环境变量 `TRAIN_PORT` 覆盖） |

---

## 方式 A：从一台「已部署好」的电脑整目录复制（最快，推荐）

原项目是**完全自包含**的：便携 Python（含 torch）、ffmpeg、RVC 引擎、GPT-SoVITS 引擎、ASR、基座模型全部在项目文件夹内。

1. 在旧机器上找到完整项目文件夹（自包含版，如 `<任意位置>\xunlianzhongxin`，作者本机约 26GB），**整个复制**到新电脑任意位置（U 盘/局域网/网盘均可）。
2. 新电脑上双击 `启动训练中心-换声.bat` 或 `启动训练中心-文字驱动.bat`，浏览器自动打开 http://127.0.0.1:8050/ 即成功。
3. （可选）把文件夹里的 `AGENTS.md / DEPLOY.md / README.md / train_service / tests / *.bat / .gitignore` 与本仓库保持同步更新。

> 如果旧机器没有完整项目、只有本仓库，请用方式 B 从零装配；也可组合使用：
> 先把本仓库 git clone / 解压到新位置，再把旧机器上的 `runtime\` `rvc\` `gptsovits\` `ziliao\` 拷进来（推荐，避免自己装配踩坑）。

---

## 方式 B：从零装配（仓库 + 官方引擎 + 官方权重）

### B1. 获取代码

```bash
git clone https://github.com/yishui111/xunlianzhongxin.git
cd xunlianzhongxin
```

### B2. 便携 Python 3.12 运行时 → `runtime\py312\`

任选其一：

- **推荐（与作者环境一致）**：从已部署机器复制整个 `runtime\py312\`。
- 自装：安装 **Python 3.12.x（64 位）** 后，用 venv/便携方式安装以下依赖（在项目根执行）：

```bash
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements_engine.txt    # 见 B5，两份引擎官方依赖合并
```

> 也可直接 `python -m venv runtime\py312` 后安装；关键是启动脚本默认找 `runtime\py312\python.exe`，
> 找不到时可设环境变量 `PY312` 指向你的 python.exe。

### B3. ffmpeg → `runtime\ffmpeg\`

下载 Windows 构建（[gyan.dev](https://www.gyan.dev/ffmpeg/builds/) 或 [BtbN](https://github.com/BtbN/FFmpeg-Builds/releases)），
把 `ffmpeg.exe`、`ffprobe.exe` 放到 `runtime\ffmpeg\bin\`。

### B4. 引擎 → `rvc\` 与 `gptsovits\GPT-SoVITS\`

```bash
# RVC（换声引擎）
git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git rvc
# GPT-SoVITS（文字驱动引擎）
git clone https://github.com/RVC-Boss/GPT-SoVITS.git gptsovits\GPT-SoVITS
```

- RVC 按官方 README 就位资产（hubert、rmvpe 等通常在其 `assets\` 或启动脚本自动下载），并在 `rvc\pretrained\` 放基座（见 B5）。
- GPT-SoVITS 需在其 `pretrained_models\` 就位 `chinese-roberta-wwm-ext-large`、`chinese-hubert-base`（官方有下载脚本/说明），顶部 `pretrained_models\` 放 s1bert/s2G/s2D 基座（见 B5）。

### B5. 基座权重（页面「下载基座模型」按钮可一键下载，也可手动放）

| 模式 | 文件 → 放置位置 | 下载 |
| ---- | ---- | ---- |
| 换声 RVC | `f0G48k.pth`、`f0D48k.pth` → `rvc\pretrained\` | https://hf-mirror.com/lj1995/VoiceConversionWebUI/tree/main/pretrained_v2 |
| 文字驱动 GPT | `s1bert.ckpt`、`s2G2333k.pth`、`s2D2333k.pth` → `gptsovits\GPT-SoVITS\pretrained_models\` | https://hf-mirror.com/lj1995/GPT-SoVITS/tree/main/gsv-v2final-pretrained |
| ASR 标字 | SenseVoiceSmall 目录 → `gptsovits\asr\SenseVoiceSmall\` | ModelScope：`iic/SenseVoiceSmall`（funasr 用目录加载） |

> 页面按钮的精确下载链接（内置在 `train_api.py` 中）：
> s1bert `https://hf-mirror.com/lj1995/GPT-SoVITS/resolve/main/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt`
> s2G / s2D：同前缀 `.../s2G2333k.pth`、`.../s2D2333k.pth`
> f0G / f0D：`https://hf-mirror.com/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G48k.pth`、`.../f0D48k.pth`

### B6. 应用 GPT-SoVITS 引擎适配补丁（3 处，必须）

以下补丁是作者在 8G~16G 显存、Windows + cu118、PyTorch 2.6 环境下实测必做的修正。
文件相对 `gptsovits\GPT-SoVITS\GPT_SoVITS\`：

**补丁 1：`s1_train.py` — 允许续训加载本机检查点（PyTorch 2.6+）**

在 `import torch` 之后（文件顶部 import 区）加入：

```python
try:
    torch.serialization.add_safe_globals([pathlib.WindowsPath])
except Exception:
    pass
```

（文件需先 `import pathlib`。原因：PyTorch 2.6+ `torch.load` 默认 `weights_only=True`，
续训/恢复加载本机检查点会报 `Unsupported global: pathlib.WindowsPath`。）

**补丁 2：`s2_train.py` — 禁用 TF32（RTX 40 系 + cu118 上 S2 起步必崩 0xC0000005）**

找到文件中设置 `torch.backends.cudnn.benchmark = False` 附近的区域，加入/改为：

```python
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
```

（原因：原启用的 TF32 低精度在 Ada 架构 + cu118 上触发 ntdll 崩溃。）

**补丁 3：`AR\data\dataset.py` — 语义文件无表头（单切片素材必现崩溃）**

在 `Text2SemanticDataset` 的 `__init__` 中找到：

```python
self.semantic_data = pd.read_csv(
    semantic_path,
    delimiter="\t",
    encoding="utf-8",
    header=None,     # ← 加上这一行
)
```

（原因：素材只有 1 个切片时 `6-name2semantic.tsv` 只有一行且无表头，
`pd.read_csv` 默认把首行当表头导致数据 0 条、训练除零崩溃。）

> 若引擎版本代码结构有差异，按「注释中的原因」对照定位即可，改动点很小。

### B7. 安装引擎依赖

按 RVC 与 GPT-SoVITS 官方 README 安装各自依赖（torch 请用 cu118 版本，与 B2 一致）。
本服务自身只需：`fastapi`、`uvicorn`、`python-multipart`、`librosa`、`soundfile`、`numpy`、`funasr`、`requests`。

### B8. 建好数据/产物目录（脚本会自动创建，也可手动建好）

```
ziliao\xunlianshuju\      # 数据集根（每个角色一个子目录，目录/文件名用拼音避免中文路径问题）
交付模型\                 # 训练完自动生成的交付包
output\                   # 训练产物存档
work\                     # 训练中间产物（自动清理）
```

### B9. 启动与验证

```bash
# 换声模式（RVC）
启动训练中心-换声.bat
# 或文字驱动模式（GPT-SoVITS）
启动训练中心-文字驱动.bat
# 统一入口：start.bat（菜单选择）；停止：关闭训练中心.bat / stop.bat
```

验证：

```
GET http://127.0.0.1:8050/api/health   → {"status":"ok","mode":"huan_sheng"|"wen_zi",...}
GET http://127.0.0.1:8050/api/env      → python/运行模式/交付目录/gpu/基座模型/数据集 是否全部正确
```

自测（可选，不跑真实训练）：

```bash
runtime\py312\python.exe tests\test_publish_clean.py   # 交付包生成 + 清理逻辑
runtime\py312\python.exe tests\test_ref_fix.py         # ref.wav 超长裁剪逻辑
```

---

## 环境变量（可覆盖默认值，全可选）

| 变量 | 默认 | 说明 |
| ---- | ---- | ---- |
| `TRAIN_MODE` | `huan_sheng` | `huan_sheng`（RVC）/ `wen_zi`（GPT-SoVITS），由启动脚本注入 |
| `TRAIN_PORT` | `8050` | Web 端口 |
| `TRAIN_AUTO_EXIT` | `1` | 训练任务全部完成后自动退出服务释放 GPU；连续训练多个角色时设 `0` |
| `PY312` | `<根>\runtime\py312\python.exe` | 便携 Python 路径覆盖 |
| `RVC_ROOT` | `<根>\rvc` | RVC 引擎根目录覆盖 |
| `GSV_ROOT` | `<根>\gptsovits\GPT-SoVITS` | GPT-SoVITS 引擎根目录覆盖 |
| `TRAIN_TARGET_STEPS` / `TRAIN_S2_TARGET_STEPS` | `8000` / `4000` | 训练量（步数）覆盖 |
| `TRAIN_S1_BATCH` / `TRAIN_S2_BATCH` | 按显存自动 | 批次覆盖 |
| `TRAIN_AUTO_PREP` | `1` | 训练前素材自动标准化开关（`0` 关闭） |

---

## 常见问题排查

- **启动提示 already running**：端口被旧实例占用 → `关闭训练中心.bat` / `stop.bat` 后重开。
- **启动提示 Portable Python not found**：`runtime\py312` 未就位或设 `PY312`。
- **`/api/env` 显示 gpu=false**：torch 未装 CUDA 版 / 驱动问题 → 重装 cu118 版 torch、更新显卡驱动。
- **页面「下载基座模型」失败**（尤其 RVC）：换 hf-mirror 直连或手动下载放对应目录。
- **训练很慢 / 显存不足**：确认未同时运行其他吃显存的服务；8G 卡系统会自动用小 batch。
- **S2 起步崩溃 0xC0000005**：确认已应用补丁 2（禁用 TF32）；并把 S2 batch 调小（16G 卡 ≤4、8G 卡 ≤2）。
- **合成漏字/音色跑偏**：交付包 ref.wav 已自动限制 3~10 秒；若仍异常检查素材质量与标字校对。
- **想换引擎/运行时位置**：设对应环境变量，不必动代码。

## 更新约定

每次优化/修复后同步更新本文件与 `README.md`、`部署方案.md`。
