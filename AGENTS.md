# 项目约定（训练中心，完全自包含）

## 项目概述

- **完全自包含**：本项目内置全部运行依赖（`runtime\py312` Python 运行时、`runtime\ffmpeg`、`rvc\` RVC 引擎+基座、`gptsovits\GPT-SoVITS` 引擎+基座、`gptsovits\asr\SenseVoiceSmall` ASR），**复制整个项目文件夹到任意电脑、任意位置，双击启动即可运行，不依赖任何外部项目/文件夹/盘符**；特殊部署可用环境变量 PY312 / RVC_ROOT / GSV_ROOT 覆盖内置路径。

模型训练中心（独立项目，端口 8050），**只负责训练，训练完自动生成交付包并清理自己**，不向任何项目自动推送：
- **换声模式**（`启动训练中心-换声.bat`，TRAIN_MODE=huan_sheng）：RVC 换音色训练 → 生成 `交付模型\rvc\<角色>\`（.pth+.index），手动复制**整个文件夹**到 换声项目 `rvc_service\models\`（目录扫描自动识别；旧方式 weights+indices 分开放也兼容）
- **文字驱动模式**（`启动训练中心-文字驱动.bat`，TRAIN_MODE=wen_zi）：GPT-SoVITS 文字驱动训练 → 生成 `交付模型\<角色>\`（ckpt+pth+ref.wav+ref_text.txt 4件套），手动复制到 文字驱动/对话模型 `tts_service\models`
- 两个项目用的模型**完全不一样**（RVC 只换音色 / GPT-SoVITS 只文字朗读），互不通用；同一角色两个功能都要用需分别训练两套模型
- 训练结束自动退出服务（TRAIN_AUTO_EXIT=1），释放主机资源
- 训练完自动生成交付包 + **自动清理该角色全部中间产物**（work 可长期保持很小）
- 素材前置处理（去BGM/去混响/去掌声/只保留主要说话人）统一由素材前置项目（8070）完成，本服务不再内置；训练直接使用处理好的素材，长音频自动切片
- 页面支持选文件夹训练：①浏览器选文件夹上传（/api/train_upload，存 work\uploaded\<角色名>\）②服务器路径直读（/api/train_dir）③默认 ziliao\xunlianshuju\<角色名>\
- **训练前自动标准化素材**（`_prep_audio`，ffmpeg）：双声道取人声声道→单声道、降噪、音量归一化到 -18dB、去长静音；产物 work\prep_<角色>\，不改原始素材；TRAIN_AUTO_PREP=0 可关闭
- 训练量默认：GPT S1 按素材时长自适应（约 100 步/分钟，500~1200 步封顶——8000 步会让 S1 在小数据集上过拟合，合成漏字；`TRAIN_GPT_TARGET_STEPS` 覆盖，旧 `TRAIN_TARGET_STEPS` 兜底兼容）+ S2 4000 步保持充足（音色像主要靠 S2，TRAIN_S2_TARGET_STEPS 可覆盖）；RVC 总步数默认 8000（`TRAIN_RVC_TARGET_STEPS` 覆盖，旧变量同样兜底）；页面 epochs 填 30 即可，自动补足
- **自动按显存选 batch**（`_auto_batches`）：≤8G→S1 4/S2 2；8-12G→6/3；≥12G→8/4（**8G 卡也能正常训练**）；TRAIN_S1_BATCH/TRAIN_S2_BATCH 可覆盖
- **S2 训练必改**：s2_train.py 已禁用 TF32（RTX40 系 + cu118 上 S2 起步必崩 0xC0000005）；S2 batch 不能大（16G 卡 4、8G 卡 2），否则显存占满卡死
- **退出时机**：队列全部任务完成（含失败）才自动退出；队列有任务/训练中绝不退出

## 关键结构

- `train_service\train_api.py`：主服务入口（FastAPI，端口 8050；TRAIN_MODE 决定模式：huan_sheng 只训 RVC / wen_zi 只训 GPT-SoVITS）。Web/API/队列/页面层；模块拆分后旧符号仍从这里导出（tests 直接 import train_api 不受影响）
- `train_service\train_common.py`：公共底座（路径常量/任务状态/进程包装执行/素材标准化/清理/下载）。依赖单向：train_common ← train_gpt / train_rvc ← train_api；跨模块可变状态（停止标志/标字校对上下文）一律走它的存取器，不要 global 裸改
- `train_service\train_gpt.py`：文字驱动训练管线（切片/ASR 标字/校对/S1/S2/ref 生成/交付）；**改文字驱动逻辑只动它**
- `train_service\train_rvc.py`：换声训练管线（切分/特征/训练/索引/推理权重提取/交付）；**改换声逻辑只动它**
- 便携 Python 是 _pth 隔离模式（sys.flags.safe_path=True）：不会自动把脚本目录加 sys.path、忽略 PYTHONPATH；train_api.py 顶部已显式插入脚本目录，子进程靠 WRAPPER/-c 注入
- `train_service\index_template.html`：页面模板（独立文件，含任务队列 UI；改页面改这里）
- `runtime\py312\`：内置 Python 3.12 运行时（含 torch，便携版）
- `runtime\ffmpeg\`：内置 ffmpeg/ffprobe
- `rvc\`：内置 RVC 引擎（含 pretrained 基座）
- `gptsovits\GPT-SoVITS\`：内置 GPT-SoVITS 引擎（含 pretrained_models 基座）
- `gptsovits\asr\SenseVoiceSmall\`：内置 ASR 标字模型
- `交付模型\<角色名>\`：训练完成自动生成的交付包（一个角色一个文件夹），**手动复制到目标项目**
- `ziliao\xunlianshuju\<角色名>\`：数据集目录（用户把目标人声放这里；目录和文件名用拼音，避免中文路径问题）；**可一键加入队列训练所有数据集目录**
- `output\<角色名>\`：训练产物存档（交付后留一份，便于回看）
- `work\`：训练中间产物（交付后自动清理）
- `tests\`：测试文件夹（test_publish_clean.py 为交付+清理自测脚本）
- `启动训练中心-换声.bat` / `启动训练中心-文字驱动.bat` / `关闭训练中心.bat`（另有总入口 `start.bat`/`stop.bat`）：一键启停（本仓库版本为**纯 ASCII 英文内容 + CRLF 换行、无 BOM**——勿写中文注释、勿存 UTF-8 BOM、勿改成 LF 换行，否则 cmd 解析会报错）。启动 bat 行为：日志实时滚黑框、服务就绪（health 200）才自动开浏览器、重复双击被端口防重拦截、**关黑框（X）即停服务释放 GPU/内存**
- `部署方案.md`：换电脑复原依据，每次优化后同步更新

## 任务队列（多角色自动依次训练）

- 页面"任务队列"卡片可添加多个角色（默认数据集目录/服务器路径/上传文件夹），或一键加入全部数据集目录
- API：`/api/train_queue`（批量加入）、`/api/train_all_datasets`（一键）、`/api/queue_status`（查询）、`/api/queue_clear`（清空等待）
- 队列任务一个接一个自动训练，**失败也继续下一个**，全部完成后按 AUTO_EXIT 自动退出
- 上传+队列：`/api/train_upload` 带 `queue=1` 只保存不训练，前端随后加入队列

## 标字校对（普通话 + 方言）

- ASR 标字后训练**自动暂停**，页面显示每段切片识别文字供核对修改（方言务必逐句校对，普通话可跳过）
- ASR 后自动丢弃可疑标注（有效字<2 或 >7 字/秒的幻觉标签，`_asr_text_ok`）：错字标签会教坏 S1 对齐，合成漏字；普通话不校对也有这道兜底
- API：`/api/review_status`（查询）、`/api/review_submit`（提交校对后继续）、`/api/review_skip`（跳过）
- 方言素材（河北话/四川话等）SenseVoice 识别错字多，必须校对后再训；这是训好方言的关键环节

## 文件归属

- 本项目数据集、产物、测试一律放本项目内，不放项目文件夹之外（网盘同步目录等）。
- 训练完只生成 `交付模型\<角色>\` 交付包，由用户手动复制到目标项目（RVC→换声项目 rvc_service\models；GPT→文字驱动/对话模型 tts_service\models），训练中心不做自动推送。

## 部署/实现任务执行方式

- 部署本项目直接执行到部署完成、测试通过再一次性汇报；不逐步请示。
- 换整体架构/方案才停下确认；小问题自行修并如实汇报。

## 工作方式（命令优先）

- 启动换声训练：双击 `启动训练中心-换声.bat`（TRAIN_MODE=huan_sheng）
- 启动文字驱动训练：双击 `启动训练中心-文字驱动.bat`（TRAIN_MODE=wen_zi）
- 停止：双击 `关闭训练中心.bat`（按端口 8050 杀进程）；训练结束也会自动退出
- 自检：`GET /api/health`（返回 mode/engine）、`GET /api/env`（返回运行模式/交付目录/GPU/基座/数据集）
- 日志：页面训练日志区；服务日志在运行窗口

## 本项目关键约定（非显而易见）

- 训练结束必须自动退出服务（TRAIN_AUTO_EXIT=1 默认开启），除非用户明确要连续训练。
- 双模式互斥：API 强制当前模式引擎，换声模式只训 RVC、文字驱动模式只训 GPT-SoVITS；两模式共用端口 8050，不要同时开。
- 模型交付路径（生成交付包后**立即清理该角色中间产物**）：
  - RVC → `交付模型\rvc\<角色名>\`（.pth + .index），用户复制**整个文件夹**到 换声项目 `rvc_service\models\<角色名>\`（旧方式 weights+indices 分开放也兼容）；清理 rvc\logs\<角色名>\
  - GPT-SoVITS → `交付模型\<角色名>\`（.ckpt + .pth + ref.wav + ref_text.txt），用户复制到 文字驱动/对话模型 `tts_service\models\<角色名>\`；清理 work 下该角色全部目录
- 交付包复制到目标项目后，目标项目目录扫描自动识别，重启/刷新即出现（无需注册）。
- 基座模型（已内置在项目内）：RVC 用 `rvc\pretrained\f0G48k.pth/f0D48k.pth`；GPT-SoVITS 用 `gptsovits\GPT-SoVITS\pretrained_models\s1bert.ckpt/s2G2333k.pth/s2D2333k.pth`。
- 文字驱动/对话模型的 `tts_service\tts_api.py` 按目录扫描读自己的 `tts_service\models`，把交付包放进去即识别。

## 禁止（Do NOT）

- 不把数据集/产物放进项目文件夹之外。
- 训练时尽量不同时跑换声/文字驱动服务（显存不够会训练失败或 OOM）。
- 不删除用户数据集（ziliao\xunlianshuju 下的内容）。
- 清理 work 时保留 `work\素材备份_*` 类目录（可能是用户手动备份的素材）。

## 维护

- 活文档：重复踩坑就补规则；每次优化/修复后同步更新 `部署方案.md`。

- RVC 基座模型（f0G48k.pth / f0D48k.pth）已就位在项目内 `rvc\pretrained\`；下载源为 hf-mirror 的 lj1995/VoiceConversionWebUI（pretrained_v2），页面按钮可一键下载。
---
### 关键点（2026-09-02 上传整理补充）
- train_service = 自研 FastAPI 训练中心（端口 8050，TRAIN_PORT 可覆盖）：RVC 换声 + GPT-SoVITS 文字驱动双模式，多角色队列/ASR 标字/按显存自动 batch
- 引擎不入库（gptsovits/、rvc/ 第三方，附官方地址）；DEPLOY.md 方式B 含 3 处 GPT-SoVITS 适配补丁说明
- 交付模型/（已训真人音色交付包）、ziliao/、output/、work/ 一律不入库；复制目标在兄弟项目（换声应用 rvc_service\models / tts 应用 tts_service\models）
- tests 自包含已跑通（test_publish_clean 5项OK、test_ref_fix 155s→8s）；README/DEPLOY 中 your-github-username 已替换 yishui111
- 三个中文名 bat 内容为纯 ASCII，勿加中文/勿改 BOM/LF
