<div align="center">

# 🎓 AI 声音训练中心（xunlianzhongxin）

> ⭐ **喜欢这个项目？请先点个 Star ⭐ 支持一下，让更多人看到！**

![GitHub stars](https://img.shields.io/github/stars/yishui111/xunlianzhongxin.svg?style=flat-square&color=orange)
![GitHub forks](https://img.shields.io/github/forks/yishui111/xunlianzhongxin.svg?style=flat-square)
![GitHub repo size](https://img.shields.io/github/repo-size/yishui111/xunlianzhongxin.svg?style=flat-square)

**给一段「目标声音」的干净说话录音，自动训练出两种声音模型：换声模型（把任意说话换成该音色）+ 文字驱动朗读模型（输入文字用该音色朗读）。**

</div>

---

## ✨ 项目简介

本中心是一个 **AI 声音模型训练服务（自研封装，FastAPI + Web 界面，端口 8050）**，把业界两大开源声音引擎 **RVC**（换音色）与 **GPT-SoVITS**（文字驱动 TTS）的整套训练流程封装成「填角色名 → 选素材文件夹 → 一键开训」的网页操作，并补齐了长音频自动切片、素材自动标准化、ASR 自动标字 + 人工校对、多角色任务队列、按显存自动选参数、训练完自动出「交付包」并清理中间产物等一整套工程化能力。

- **换声模式**（`启动训练中心-换声.bat`）：RVC 换音色训练。适合「换声」类应用 —— 别人说的话保留语句/时长，只把音色换成目标声音。
- **文字驱动模式**（`启动训练中心-文字驱动.bat`）：GPT-SoVITS 文字驱动训练。适合「文字朗读/TTS」类应用 —— 输入文字，用目标音色朗读出来。
- 两种模型**互不通用**（RVC 只换音色、GPT-SoVITS 只文字朗读），同一角色两种功能都要用，需分别训练两套模型（素材可共用）。
- 训练完成后自动生成 **`交付模型\<角色名>\`** 交付包，由你**手动复制**到下游应用（换声应用 / 文字驱动应用）的模型目录即可被自动识别 —— 训练中心不做任何自动推送，各项目完全解耦。

> 素材准备建议：3~10 分钟以上、干净的单人说话录音（去 BGM/去混响/去掌声等「内容级处理」由素材前置项目负责，本中心负责切片与「标准级处理」：单声道、降噪、音量归一化、去长静音）。

## 🎯 主要功能

- 🎤 **两种训练模式**：RVC 换音色 / GPT-SoVITS 文字驱动，同一 Web 服务（8050），启动脚本决定模式，API 强制互斥
- 📂 **三种素材来源**：浏览器选文件夹上传 / 服务器路径直读 / 默认数据集目录 `ziliao\xunlianshuju\<角色名>\`
- ✂️ **自动预处理**：长音频自动切片（超长段二次切分）、单声道/降噪/音量归一化(-18dB)/去长静音，不改原始素材
- 📝 **ASR 标字 + 人工校对**：SenseVoice 自动标字后暂停，页面逐句核对修改（方言素材必校，普通话可跳过）
- 📋 **任务队列**：多角色排队自动依次训练，任一失败自动继续下一个，全部完成后自动退出释放 GPU
- 🧠 **按显存自适应**：≤8G→S1 4/S2 2、8~12G→6/3、≥12G→8/4，**8G 显卡也能正常训练**
- 📦 **交付包 + 自动清理**：训练完生成 `交付模型\<角色>\`（换声 .pth+.index / 文字驱动 4件套），并立即清掉该角色全部中间产物
- 🔌 **完全自包含**：运行时/引擎/基座全部按项目内目录布局，路径随脚本自动推导，环境变量可覆盖（PY312 / RVC_ROOT / GSV_ROOT / TRAIN_PORT 等）

## 🗂️ 目录结构

**仓库内容（仅源码/脚本/配置/文档）：**

```
xunlianzhongxin/
├── train_service/
│   ├── train_api.py             # 主服务（FastAPI，端口 8050，TRAIN_MODE 双模式）
│   └── index_template.html      # Web 页面模板（任务队列/标字校对/日志/环境检查）
├── tests/
│   ├── test_publish_clean.py    # 交付包生成 + 中间产物清理自测
│   └── test_ref_fix.py          # GPT 参考音频（ref.wav）超长裁剪自测
├── 启动训练中心-换声.bat          # 换声模式（RVC）启动
├── 启动训练中心-文字驱动.bat       # 文字驱动模式（GPT-SoVITS）启动
├── 关闭训练中心.bat              # 一键停止（按端口 8050）
├── start.bat / stop.bat         # 启停总入口
├── AGENTS.md                    # 项目约定（给 AI 代理看的规则）
├── DEPLOY.md                    # 新电脑部署步骤（必读）
├── 部署方案.md                   # 架构与运行方案说明
├── README.md                    # 本文件
└── .gitignore
```

**部署到新电脑后的完整运行目录（引擎/运行时为「去下载/复制」的大件，见下方下载表）：**

```
xunlianzhongxin/                  # = 上面仓库 + 下面这些目录
├── runtime\py312\                # 便携 Python 3.12 运行时（含 torch）
├── runtime\ffmpeg\               # ffmpeg/ffprobe
├── rvc\                          # RVC 引擎（含 rvc\pretrained\ 基座）
├── gptsovits\GPT-SoVITS\         # GPT-SoVITS 引擎（含 pretrained_models\ 基座）
├── gptsovits\asr\SenseVoiceSmall\  # ASR 标字模型
├── ziliao\xunlianshuju\<角色名>\   # 数据集目录（放目标人声）
├── 交付模型\<角色名>\              # 训练完自动生成的交付包（复制到下游应用）
├── output\<角色名>\               # 训练产物存档
└── work\                         # 训练中间产物（交付后自动清理）
```

> 💡 模型权重、素材、第三方引擎、运行时等**大文件不随仓库分发**，见下方「大件资源下载」与 `DEPLOY.md`。

## 🚀 快速开始（拉到新电脑即可部署）

> ⚠️ 本仓库不含引擎/运行时/基座权重（体积巨大）。首次部署请按 **[DEPLOY.md](DEPLOY.md)** 执行（推荐「方式 A：从一台已部署机器整目录复制」），把下列目录就位后再启动：
> `runtime\`、`rvc\`、`gptsovits\`、`ziliao\`、`交付模型\`、`output\`、`work\`

### 环境要求

- 操作系统：Windows 10/11（bat 一键启动）
- 显卡：NVIDIA GPU（训练必须 GPU，显存建议 ≥ 8GB，训练参数按显存自动适配）
- 磁盘：数据集 + 训练产物预留 20GB 以上

### 1. 克隆

```bash
git clone https://github.com/yishui111/xunlianzhongxin.git
cd xunlianzhongxin
```

### 2. 就位引擎/运行时

按 `DEPLOY.md` 把 `runtime`、`rvc`、`gptsovits` 及基座模型放到对应目录（或从已部署机器整个复制）。

### 3. 启动

```bash
# Windows：按需双击（或命令行执行）：
#  换声模式（RVC）        → 启动训练中心-换声.bat   （或 start.bat 选 1）
#  文字驱动模式（GPT-SoVITS）→ 启动训练中心-文字驱动.bat （或 start.bat 选 2）
#  停止                  → 关闭训练中心.bat / stop.bat
start.bat
```

### 4. 验证

打开浏览器访问 http://127.0.0.1:8050/ ，页面正常显示即部署成功。也可直接验证：

- `GET /api/health` → `{"status":"ok","mode":"huan_sheng","engine":"rvc"}`（或 wen_zi/gpt）
- `GET /api/env` → 显示运行模式/GPU 显存/批次/基座模型/数据集等

## 🎮 使用流程（训练一个角色）

1. 双击对应模式启动脚本，浏览器自动打开训练页面（http://127.0.0.1:8050/）
2. 填角色名（拼音/英文，如 `juese_a`），选素材来源（上传文件夹 / 服务器路径 / 放入 `ziliao\xunlianshuju\<角色名>\`）
3. epochs 填 30 即可（系统自动按目标步数补足）；方言素材勾选「需要标字校对」
4. 点「开始训练」或加入任务队列，页面实时显示日志
5. 训练完自动生成 **`交付模型\<角色名>\`** 并清理中间产物，服务自动退出
6. **手动把交付包复制到下游应用**：换声模型 → 换声应用 `rvc\assets`；文字驱动模型 4件套 → 文字驱动应用 `tts_service\models\<角色名>\`

## 📥 大件资源下载（模型 / 引擎 / 运行时）

| 资源 | 位置（就位到） | 下载地址 / 获取方式 |
| ---- | ---- | ---- |
| RVC 引擎 | `rvc\` | 官方开源：[RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)（按官方 README 准备 hubert/rmvpe 等资产） |
| GPT-SoVITS 引擎 | `gptsovits\GPT-SoVITS\` | 官方开源：[RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) |
| RVC 基座模型 f0G48k/f0D48k | `rvc\pretrained\` | [hf-mirror lj1995/VoiceConversionWebUI（pretrained_v2）](https://hf-mirror.com/lj1995/VoiceConversionWebUI/tree/main/pretrained_v2)；页面「下载基座模型」按钮可一键下载 |
| GPT-SoVITS 基座 s1bert/s2G/s2D | `gptsovits\GPT-SoVITS\pretrained_models\` | [hf-mirror lj1995/GPT-SoVITS（gsv-v2final-pretrained）](https://hf-mirror.com/lj1995/GPT-SoVITS/tree/main/gsv-v2final-pretrained)；页面按钮可一键下载 |
| SenseVoiceSmall ASR 模型 | `gptsovits\asr\SenseVoiceSmall\` | ModelScope：`iic/SenseVoiceSmall`（funasr 用目录加载） |
| 便携 Python 3.12 运行时（含 torch cu118） | `runtime\py312\` | 自装 Python 3.12 + `pip install torch --index-url https://download.pytorch.org/whl/cu118` 及依赖（完整清单见 DEPLOY.md） |
| ffmpeg | `runtime\ffmpeg\` | 官方构建（[gyan.dev](https://www.gyan.dev/ffmpeg/builds/) / BtbN） |

> 详细步骤与「引擎适配补丁」见 [DEPLOY.md](DEPLOY.md)。

## 🛠️ 本地开发 & 提交

```bash
git add .
git commit -m "feat: xxx"
git push origin main
```

> 注意 `.gitignore` 已把 `runtime/` `rvc/` `gptsovits/` `ziliao/` `work/` `output/` `交付模型/` 等重型/敏感目录排除，放心本地使用，不会误提交。

## ❓ 常见问题（FAQ）

- **Q：双击启动提示「Train Center is already running」？** A：端口 8050 已有训练中心实例，先双击 `关闭训练中心.bat` 或 `stop.bat` 停掉旧实例再启动。
- **Q：提示 [ERROR] Portable Python not found？** A：`runtime\py312` 未就位，按 `DEPLOY.md` 补齐运行时/引擎后再启动。
- **Q：没有 NVIDIA 显卡能训练吗？** A：不能（训练依赖 GPU）；无 GPU 只能启动页面但训练会失败。
- **Q：训练出来读错字/漏字？** A：多为 ASR 标字错误，方言素材务必勾选「需要标字校对」逐句核对；普通话识别较准可跳过。
- **Q：参考音频 ref.wav 要多长？** A：交付包已自动保证 3~10 秒（超长自动裁剪到 8 秒），无需手工处理。
- **Q：同一个角色想换声和文字驱动都要？** A：需分别用两个模式各训一套模型，素材可共用。
- **Q：两个启动脚本能同时开吗？** A：不能，共用端口 8050 且训练需独占 GPU/显存，请一次只开一个模式。

## ⚠️ 注意事项

- 训练用的素材请使用**你有权使用的声音**；请勿对他人声音进行未授权的克隆与传播。
- 敏感信息（密钥、token、账号密码）一律放 `.env` 或环境变量，禁止提交到仓库。
- 本仓库只包含源代码/脚本/配置/文档；第三方引擎、运行时、模型权重均为其各自作者的产物，请遵守其开源协议。
- 本仓库仅供学习交流使用。

## 📄 许可证

MIT License（如仓库内后续放置 LICENSE 文件则以仓库内为准）

## 🙏 支持与致谢

如果这个项目帮到了你，**请点亮右上角的 ⭐ Star**，你的支持是我持续更新的最大动力！
