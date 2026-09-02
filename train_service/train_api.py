# -*- coding: utf-8 -*-
"""
训练中心（独立项目，端口 8050）— 模型训练服务（完全自包含）
==============================================================
两种训练模式（TRAIN_MODE）：
  huan_sheng ：RVC 换音色训练 → 生成交付包（交付模型\\rvc\\<角色>\\，.pth + .index）
  wen_zi     ：GPT-SoVITS 文字驱动 TTS 训练 → 生成交付包（交付模型\\<角色>\\，ckpt+pth+ref.wav+ref_text.txt）

本项目完全自包含：运行时（runtime\\py312）、ffmpeg（runtime\\ffmpeg）、
训练引擎（rvc\\、gptsovits\\GPT-SoVITS）、基座模型全部内置在项目内，
复制整个项目文件夹到任意电脑、任意位置，双击启动脚本即可运行，
不依赖任何外部项目/文件夹。

训练完成只生成"交付模型"文件夹，不自动推送任何项目；
把 交付模型\\<角色>\\ 整个文件夹复制到目标项目的 models 目录即可被自动识别。
特殊部署可用环境变量 PY312 / RVC_ROOT / GSV_ROOT 覆盖内置路径。

运行（内置 Python 3.12）：
  runtime\\py312\\python.exe train_service\\train_api.py
"""

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
import urllib.request

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# 项目根目录：随脚本位置自动推导（train_service 的上一级），复制到任意位置都能用
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============ 完全自包含：运行时/引擎/基座全部在项目内，不依赖外部任何文件夹 ============
# 内置 Python 运行时（runtime\py312，便携版，直接复制即用）
PY312 = os.environ.get("PY312") or os.path.join(PROJECT_ROOT, "runtime", "py312", "python.exe")
# 内置 RVC 引擎（rvc\，含 configs + pretrained 基座）
RVC_ROOT = os.environ.get("RVC_ROOT") or os.path.join(PROJECT_ROOT, "rvc")
# 内置 GPT-SoVITS 引擎（gptsovits\GPT-SoVITS，含 pretrained_models 基座）
GSV_ROOT = os.environ.get("GSV_ROOT") or os.path.join(PROJECT_ROOT, "gptsovits", "GPT-SoVITS")

# ================= 运行模式（解耦：训练中心只训练，只产出"交付模型"文件夹） =================
# TRAIN_MODE=huan_sheng（默认）→ 换声模型训练中心：只训 RVC，交付包 交付模型\rvc\<角色>\
# TRAIN_MODE=wen_zi            → 文字驱动模型训练中心：只训 GPT-SoVITS，交付包 交付模型\<角色>\
TRAIN_MODE = os.environ.get("TRAIN_MODE", "huan_sheng")
MODE_ENGINE = "rvc" if TRAIN_MODE == "huan_sheng" else "gpt"
MODE_NAME = {"huan_sheng": "换声模型训练中心", "wen_zi": "文字驱动模型训练中心"}.get(TRAIN_MODE, "训练中心")
MODE_DESC = {
    "huan_sheng": (
        "· <b>RVC 换音色训练</b>：训练某个人的音色，之后把其他人说的话换成这个音色（语句/时长保留）"
        + "→ 生成<b>交付模型</b>文件夹（" + os.path.join(PROJECT_ROOT, "交付模型", "rvc") + "，复制到换声项目 rvc\\assets 即可用）<br>"
        + "· 训练前自动标准化素材（音量归一化/降噪/去长静音/单声道），录音音量小也没关系<br>"
        + "· 本模式只训练换声模型（RVC），训练完自动生成交付包并清理中间产物，释放空间"),
    "wen_zi": (
        "· <b>GPT-SoVITS 文字驱动训练</b>：训练后支持“文字→语音”（TTS）"
        + "→ 生成<b>交付模型</b>文件夹（" + os.path.join(PROJECT_ROOT, "交付模型") + "，复制到 文字驱动/对话模型 的 tts_service\\models 即可用）<br>"
        + "· 训练前自动标准化素材（音量归一化/降噪/去长静音/单声道），录音音量小也没关系<br>"
        + "· 本模式只训练文字驱动模型（GPT-SoVITS），训练完自动生成交付包并清理中间产物，释放空间"),
}.get(TRAIN_MODE, "")

# 便携 Python 隔离模式忽略 PYTHONPATH，用 -c 包装注入引擎根目录
GSV_PYTHONPATH = os.pathsep.join([GSV_ROOT, os.path.join(GSV_ROOT, "GPT_SoVITS")])
# pythonpath 支持多个路径（os.pathsep 分隔）：脚本同时依赖顶层 tools 和 GPT_SoVITS 内部模块
# 模拟 `python 脚本.py` 的标准行为：脚本所在目录也加入 sys.path（tools/slicer2 等按脚本目录解析）
WRAPPER = ("import os,sys; [sys.path.insert(0,p) for p in reversed(sys.argv[1].split(os.pathsep)) if p]; "
           "sys.argv=sys.argv[2:]; sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0]))); "
           "exec(open(sys.argv[0],encoding='utf-8').read())")
DATA_ROOT = os.path.join(PROJECT_ROOT, "ziliao", "xunlianshuju")
WORK_ROOT = os.path.join(PROJECT_ROOT, "work")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output")
# 交付模型目录：训练完成只在这里生成交付包（一个角色一个文件夹），
# 由用户手动复制到 文字驱动/对话模型 的 tts_service\models 或 换声项目 rvc\assets
DELIVERY_ROOT = os.path.join(PROJECT_ROOT, "交付模型")
RVC_DELIVERY_DIR = os.path.join(DELIVERY_ROOT, "rvc")
PORT = int(os.environ.get("TRAIN_PORT", "8050"))
# 训练结束自动退出（默认开），释放主机资源；需要连续训练多个模型时置 0
AUTO_EXIT = os.environ.get("TRAIN_AUTO_EXIT", "1") == "1"
_server = None
# 当前训练子进程与手动停止标志（供"停止训练"：杀进程树 + 标记停止，队列自动继续下一个或退出）
_active_proc = None
_stop_requested = False

# 保证 ffmpeg/ffprobe 可用（不依赖启动脚本设置 PATH，直接 python 启动也能训练/切分/测时长）
_FFMPEG_BIN = os.path.join(PROJECT_ROOT, "runtime", "ffmpeg", "bin")
if os.path.isdir(_FFMPEG_BIN):
    os.environ["PATH"] = _FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")

RVC_BASE = {
    "G": os.path.join(RVC_ROOT, "pretrained", "f0G48k.pth"),
    "D": os.path.join(RVC_ROOT, "pretrained", "f0D48k.pth"),
}
GPT_BASE = {
    # 基座模型实际放在 GPT-SoVITS\pretrained_models\（顶层），不在 GPT_SoVITS\pretrained_models\
    "s1bert": os.path.join(GSV_ROOT, "pretrained_models", "s1bert.ckpt"),
    "s2G": os.path.join(GSV_ROOT, "pretrained_models", "s2G2333k.pth"),
    "s2D": os.path.join(GSV_ROOT, "pretrained_models", "s2D2333k.pth"),
}
GPT_DL = {
    "s1bert": "https://hf-mirror.com/lj1995/GPT-SoVITS/resolve/main/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
    "s2G": "https://hf-mirror.com/lj1995/GPT-SoVITS/resolve/main/gsv-v2final-pretrained/s2G2333k.pth",
    "s2D": "https://hf-mirror.com/lj1995/GPT-SoVITS/resolve/main/gsv-v2final-pretrained/s2D2333k.pth",
}
RVC_DL = {
    "G": "https://hf-mirror.com/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G48k.pth",
    "D": "https://hf-mirror.com/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D48k.pth",
}


os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(WORK_ROOT, exist_ok=True)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("train_center")

# ---------------- 任务状态 ----------------
_lock = threading.Lock()
_state = {
    "running": False,
    "engine": "",
    "name": "",
    "step": "",
    "ok": False,
    "error": "",
    "log": [],
}

# ---------------- 任务队列（多角色排队依次训练） ----------------
# 队列任务：{"engine", "name", "epochs", "clean", "raw_dir", "status": waiting/running/done/failed, "result"}
_queue_lock = threading.Lock()
_queue = []
_queue_done = []

# ---------------- 标字校对（ASR 识别后暂停，等用户核对/修改再继续训练） ----------------
# 普通话识别准确可"跳过校对"；方言（河北话/四川话等）务必逐句校对后再训练
_review_evt = threading.Event()
_review_pending = {}  # {"name", "wavs":[完整路径], "items":[{"file","text"}], "inp_text"}


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    with _lock:
        _state["log"].append(line)
        _state["log"] = _state["log"][-800:]
    logger.info(msg)


def set_state(**kw):
    with _lock:
        _state.update(kw)


_exit_armed = False


def schedule_exit(delay=5):
    """所有任务（含队列）全部完成后才自动退出，释放 GPU/内存。
    队列还有任务、或任务正在运行、或退出已排程时，不会重复安排。"""
    global _exit_armed
    if not AUTO_EXIT:
        return
    with _queue_lock:
        if _queue:
            return  # 还有排队任务，不退出
    with _lock:
        if _state["running"]:
            return  # 还有任务在跑，不退出
        if _exit_armed:
            return
        _exit_armed = True

    def _do():
        time.sleep(delay)
        # 退出前再复查：有新任务加入则取消退出
        with _queue_lock:
            if _queue:
                return
        with _lock:
            if _state["running"]:
                return
        log("全部训练任务已完成，训练中心将在 %d 秒后自动退出（释放主机资源）" % delay)
        if _server is not None:
            _server.should_exit = True
        else:
            os._exit(0)
    threading.Thread(target=_do, daemon=True).start()


def _cancel_exit():
    """新任务开始时调用：取消已排程的退出。"""
    global _exit_armed
    _exit_armed = False


def run_proc(cmd, cwd, step, pythonpath=None):
    global _active_proc
    log("▶ %s" % step)
    log("  %s" % " ".join(str(c) for c in cmd))
    if _stop_requested:
        raise RuntimeError("训练已手动停止")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    if pythonpath:
        env["PYTHONPATH"] = pythonpath + os.pathsep + env.get("PYTHONPATH", "")
    if pythonpath and cmd and str(cmd[0]) == PY312 and str(cmd[1]).lower().endswith(".py"):
        cmd = [PY312, "-c", WRAPPER, pythonpath, str(cmd[1])] + [str(x) for x in cmd[2:]]
    p = subprocess.Popen(
        [str(c) for c in cmd],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        bufsize=1,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    _active_proc = p
    try:
        for line in p.stdout:
            line = line.rstrip("\n\r")
            if line:
                log(line[:300])
        p.wait()
    finally:
        _active_proc = None
    if p.returncode != 0:
        raise RuntimeError("%s 失败（exit=%s）" % (step, p.returncode))


def list_audio(path):
    if os.path.isfile(path):
        return [path] if path.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg")) else []
    if not os.path.isdir(path):
        return []
    out = []
    for f in os.listdir(path):
        if f.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg")):
            out.append(os.path.join(path, f))
    return out


# ---------------- 素材前置处理（去 BGM / 去混响） ----------------
def train_rvc(name, epochs, clean=True, raw_dir=None):
    exp = os.path.join(RVC_ROOT, "logs", name)
    raw = raw_dir or os.path.join(DATA_ROOT, name)
    # 训练前自动素材标准化（音量/降噪/去长静音/单声道，不改原始素材）
    raw = _prep_audio(raw, name)
    audios = list_audio(raw)
    if not audios:
        raise RuntimeError("数据集目录 %s 没有音频，请先把目标人声放入该目录" % raw)
    if clean:
        log("素材前置处理（去BGM/去混响/去掌声）由素材前置项目（8070）完成，本服务直接使用素材训练")

    set_state(step="准备数据集")
    log("数据集音频：%d 个文件（共 %.1f 秒）" % (
        len(audios), sum(_audio_dur(a) for a in audios)))
    if os.path.isdir(exp):
        shutil.rmtree(exp, ignore_errors=True)
    os.makedirs(exp, exist_ok=True)
    shutil.copy(
        os.path.join(RVC_ROOT, "configs", "v2", "48k.json"),
        os.path.join(exp, "config.json"),
    )

    set_state(step="1/6 数据切分")
    run_proc([PY312, "-m", "train.preprocess",
              raw, "48000", "8", exp, "False", "3.7"], RVC_ROOT, "RVC 数据切分", pythonpath=RVC_ROOT)

    set_state(step="2/6 提取 F0")
    run_proc([PY312, "-m", "train.dataset.extract_f0",
              "cuda", "1", "0", "0", exp, "False"], RVC_ROOT, "RVC F0 提取", pythonpath=RVC_ROOT)

    set_state(step="3/6 提取 HuBERT 特征")
    run_proc([PY312, "-m", "train.dataset.extract_hubert_feature",
              "cuda", "1", "0", exp, "v2", "False"], RVC_ROOT, "RVC HuBERT 特征", pythonpath=RVC_ROOT)

    set_state(step="生成训练清单")
    wavs = sorted(glob_wavs(os.path.join(exp, "1_16k_wavs")))
    if not wavs:
        raise RuntimeError("切分后没有生成训练音频，请检查素材是否清晰人声")
    with open(os.path.join(exp, "filelist.txt"), "w", encoding="utf-8") as f:
        for w in wavs:
            base = os.path.splitext(os.path.basename(w))[0]
            f.write("%s|%s|%s|%s|0\n" % (
                os.path.join(exp, "0_gt_wavs", os.path.basename(w)),
                os.path.join(exp, "3_feature768", base + ".npy"),
                os.path.join(exp, "2a_f0", base + ".wav.npy"),
                os.path.join(exp, "2b-f0nsf", base + ".wav.npy"),
            ))
    log("训练清单：%d 条" % len(wavs))

    # 训练量自动补足：RVC 官方建议总步数约 10000~20000，按素材时长自适应（防过拟合）
    # batch 按显卡显存自适应（8G 卡用小 batch，防显存不足）
    batch_size = _auto_batches()[1]
    total_sec = sum(_audio_dur(a) for a in audios)
    target_steps = _target_steps_by_data(total_sec)
    cur_steps = len(wavs) * epochs / batch_size
    if cur_steps < target_steps:
        auto_epochs = max(epochs, (target_steps * batch_size + len(wavs) - 1) // len(wavs))
        log("训练量自动补足：切片 %d × epochs %d / batch %d = %d 步，未达目标 %d 步，自动提高到 epochs=%d（约 %d 步）" % (
            len(wavs), epochs, batch_size, int(cur_steps), target_steps,
            auto_epochs, auto_epochs * len(wavs) // batch_size))
        epochs = auto_epochs
    else:
        log("训练量确认：切片 %d × epochs %d / batch %d = %d 步（目标 %d 步）" % (
            len(wavs), epochs, batch_size, int(cur_steps), target_steps))

    set_state(step="4/6 训练模型（epochs=%d）" % epochs)
    cmd = [PY312, "-m", "train.train",
           "-e", name, "-sr", "48k", "-v", "v2", "-f0", "1",
           "-bs", str(batch_size), "-g", "0", "-te", str(epochs), "-se", "5",
           "-l", "1", "-c", "0"]
    if os.path.isfile(RVC_BASE["G"]) and os.path.isfile(RVC_BASE["D"]):
        cmd += ["-pg", RVC_BASE["G"], "-pd", RVC_BASE["D"]]
        log("使用预训练基座模型（更高质量）")
    else:
        log("未找到预训练基座模型，将从零训练（质量较低，建议先下载基座模型）")
    run_proc(cmd, RVC_ROOT, "RVC 训练", pythonpath=RVC_ROOT)

    set_state(step="5/6 训练索引并生成交付包")
    g_file = latest_glob(os.path.join(exp, "G_*.pth"))
    if not g_file or not os.path.isfile(g_file):
        raise RuntimeError("训练完成但没找到模型权重（logs 下无 G_*.pth）")
    out_dir = os.path.join(OUTPUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(g_file, os.path.join(out_dir, name + ".pth"))
    run_proc([PY312, "-m", "train.train_index",
              name, "v2", "assets/indices", "8"], RVC_ROOT, "RVC 索引训练", pythonpath=RVC_ROOT)
    # 生成交付包：交付模型\rvc\<角色>\（<角色>.pth + <角色>.index），用户手动复制到换声项目 rvc\assets
    deliv = os.path.join(RVC_DELIVERY_DIR, name)
    os.makedirs(deliv, exist_ok=True)
    shutil.copy(g_file, os.path.join(deliv, name + ".pth"))
    idx = os.path.join(RVC_ROOT, "assets", "indices", name + ".index")
    if os.path.isfile(idx):
        shutil.copy(idx, os.path.join(deliv, name + ".index"))
    log("模型已交付：%s → 交付模型\\rvc\\%s（%s），复制该文件夹到 换声项目 rvc\\assets 即可使用" % (name, name, deliv))
    _clean_work(name, "rvc")


def _reslice_long(sliced, max_sec=12, target_sec=8):
    """把超长切片（>max_sec 秒）二次切成 target_sec 秒固定段。
    切片脚本按静音切，素材连续说话无停顿时会切出超长段（60~96 秒），
    导致训练特征巨大、计算爆炸、训练卡死。二次切分是标准做法，不影响质量。"""
    import soundfile as sf
    cut = 0
    removed = 0
    for w in list_audio(sliced):
        try:
            info = sf.info(w)
            dur = info.frames / info.samplerate
            if dur <= max_sec:
                continue
            y, sr = sf.read(w, dtype="float32")
            base = os.path.splitext(os.path.basename(w))[0]
            step = int(target_sec * sr)
            idx = 0
            total = len(y)
            while idx * step < total:
                seg = y[idx * step:(idx + 1) * step]
                if len(seg) < int(2.0 * sr):  # 尾部不足 2 秒丢弃（太短无训练价值）
                    break
                sf.write(os.path.join(sliced, "%s_s%d.wav" % (base, idx)), seg, sr)
                idx += 1
            if idx > 0:
                os.remove(w)
                removed += 1
                cut += idx
                log("超长切片 %s（%.0f 秒）→ 切成 %d 段" % (os.path.basename(w), dur, idx))
        except Exception:
            pass
    if removed:
        log("已把 %d 个超长切片二次切分为 %d 个 %d 秒段（训练数据时长已正常）"
            % (removed, cut, target_sec))
    return cut


def _audio_dur(path):
    # ffprobe 最快且支持 wav/mp3/m4a/aac/flac/ogg
    try:
        import subprocess
        ffprobe = os.path.join(os.path.dirname(PY312), "..", "ffmpeg", "bin", "ffprobe.exe")
        r = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=20)
        if r.returncode == 0 and r.stdout.strip():
            return float(r.stdout.strip())
    except Exception:
        pass
    try:
        import soundfile as sf
        return sf.info(path).frames / sf.info(path).samplerate
    except Exception:
        try:
            import librosa
            return librosa.get_duration(path=path)
        except Exception:
            return 0.0


def _target_steps_by_data(total_sec):
    """S1 目标步数：默认 8000 步（10 分钟左右素材的社区推荐量，质量与耗时平衡；
    素材更短可 TRAIN_TARGET_STEPS 覆盖调小防过拟合）。"""
    return int(os.environ.get("TRAIN_TARGET_STEPS", "8000"))


def _prep_audio(raw_dir, name, label="素材标准化"):
    """训练前自动素材标准化（不修改原始素材，产物放 work/prep_<name>/）：
    1) 双声道自动取人声声道（左右响度大的那个）→ 单声道
    2) 高通/低通滤波 + 轻降噪（afftdn）
    3) 音量归一化到约 -18dB（动态算增益 + 限幅防削波）
    4) 去开头/结尾及中间 >2 秒长静音
    统一由 ffmpeg 完成；素材已是干净录音也可设 TRAIN_AUTO_PREP=0 关闭。"""
    if os.environ.get("TRAIN_AUTO_PREP", "1") != "1":
        log("TRAIN_AUTO_PREP=0：跳过自动素材标准化")
        return raw_dir
    prep = os.path.join(WORK_ROOT, "prep_" + name)
    if os.path.isdir(prep):
        shutil.rmtree(prep, ignore_errors=True)
    os.makedirs(prep, exist_ok=True)
    ffb = os.path.join(os.path.dirname(PY312), "..", "ffmpeg", "bin")
    ffmpeg = os.path.join(ffb, "ffmpeg.exe")
    ffprobe = os.path.join(ffb, "ffprobe.exe")
    audios = list_audio(raw_dir)
    if not audios:
        return raw_dir
    log("▶ %s：自动处理 %d 个音频（单声道/降噪/音量归一化/去长静音），不改原始素材" % (label, len(audios)))
    done = 0
    for a in audios:
        out = os.path.join(prep, os.path.splitext(os.path.basename(a))[0] + ".wav")
        nch = 1
        try:
            r = subprocess.run([ffprobe, "-v", "error", "-select_streams", "a:0",
                                "-show_entries", "stream=channels", "-of", "csv=p=0", a],
                               capture_output=True, text=True, timeout=30)
            if r.stdout.strip().isdigit():
                nch = int(r.stdout.strip())
        except Exception:
            pass
        ch = 0
        if nch >= 2:
            # 双声道：比较左右响度，取响度大的（人声声道）
            vols = []
            for c in range(2):
                try:
                    r2 = subprocess.run([ffmpeg, "-i", a, "-af", "pan=mono|c0=c%d,volumedetect" % c,
                                         "-f", "null", "NUL"], capture_output=True, text=True, timeout=180)
                    m = re.search(r"mean_volume:\s*(-?[\d.]+) dB", r2.stderr)
                    vols.append((float(m.group(1)) if m else 0.0, c))
                except Exception:
                    vols.append((0.0, c))
            ch = max(vols)[1]
        gain = 8.0
        try:
            r3 = subprocess.run([ffmpeg, "-i", a, "-af", "volumedetect", "-f", "null", "NUL"],
                                capture_output=True, text=True, timeout=180)
            m = re.search(r"mean_volume:\s*(-?[\d.]+) dB", r3.stderr)
            if m:
                gain = max(0.0, min(-18.0 - float(m.group(1)), 20.0))
        except Exception:
            pass
        filters = []
        if nch >= 2:
            filters.append("pan=mono|c0=c%d" % ch)
        filters.append("highpass=f=80,lowpass=f=8000")
        filters.append("afftdn=nf=-30")
        filters.append("volume=%.1fdB" % gain)
        filters.append("alimiter=limit=0.95")
        filters.append("silenceremove=start_periods=1:start_threshold=-45dB:start_duration=1.5"
                       ":stop_periods=-1:stop_threshold=-45dB:stop_duration=2")
        cmd = [ffmpeg, "-y", "-i", a, "-af", ",".join(filters),
               "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", out]
        try:
            r4 = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if r4.returncode == 0 and os.path.isfile(out) and os.path.getsize(out) > 0:
                done += 1
            else:
                shutil.copy(a, out)  # 处理失败保底：用原文件
        except Exception:
            shutil.copy(a, out)
    log("%s：完成 %d/%d 个，产物 %s" % (label, done, len(audios), prep))
    return prep


def estimate_training(folder):
    """根据素材文件夹估算：总时长、预计切片数、建议 epochs、预计训练时长。
    经验值：RVC 切分后有效音频约占 65%；RTX 4080 每步约 0.35 秒；
    前置处理约每 60 秒素材耗 1.5 秒；切分/F0/HuBERT 固定约 90 秒。"""
    audios = list_audio(folder)
    if not audios:
        return None
    total_sec = sum(_audio_dur(a) for a in audios)
    slices = max(1, int(total_sec * 0.65 / 3.7))
    batch = 4
    target_steps = _target_steps_by_data(total_sec)
    need_epochs = max(1, (target_steps * batch + slices - 1) // slices)
    train_min = target_steps * 0.35 / 60
    pre_min = total_sec * 1.5 / 60 / 60 + 0.2
    fixed_min = 1.5
    total_min = train_min + pre_min + fixed_min
    return {
        "audio_files": len(audios),
        "total_sec": round(total_sec, 1),
        "total_min": round(total_sec / 60, 1),
        "est_slices": slices,
        "target_steps": target_steps,
        "need_epochs": need_epochs,
        "est_train_min": round(train_min, 1),
        "est_pre_min": round(pre_min, 1),
        "est_total_min": round(total_min, 1),
    }


def glob_wavs(d):
    import glob
    return sorted(glob.glob(os.path.join(d, "*.wav")))


def latest_glob(pattern):
    import glob
    files = glob.glob(pattern)
    if not files:
        return None
    def key(f):
        m = re.search(r"G_(\d+)\.pth$", f)
        return int(m.group(1)) if m else 0
    return max(files, key=key)


# ---------------- GPT-SoVITS 训练 ----------------
def train_gpt(name, epochs, clean=True, raw_dir=None, review=False):
    raw = raw_dir or os.path.join(DATA_ROOT, name)
    # 训练前自动素材标准化（音量/降噪/去长静音/单声道，不改原始素材）
    raw = _prep_audio(raw, name)
    audios = list_audio(raw)
    if not audios:
        raise RuntimeError("数据集目录 %s 没有音频" % raw)
    if clean:
        log("素材前置处理（去BGM/去混响/去掌声）由素材前置项目（8070）完成，本服务直接使用素材训练")

    work = os.path.join(WORK_ROOT, "gpt_" + name)
    opt_dir = os.path.join(work, "fine_tune_dataset", name)
    s2_ckpt_dir = os.path.join(opt_dir, "logs_s2_v2")
    # 续训支持：已有 S2 检查点（logs_s2_v2/G_*.pth）时跳过预处理和 S1，直接续训 S2（s2_train.py 自动 resume）
    s2_ckpt_files = [f for f in os.listdir(s2_ckpt_dir)
                     if f.startswith("G_") and f.endswith(".pth")] if os.path.isdir(s2_ckpt_dir) else []
    if s2_ckpt_files:
        import json as _json
        try:
            with open(os.path.join(work, "s2_%s.json" % name), encoding="utf-8") as f:
                s2_epochs = int(_json.load(f)["train"]["epochs"])
        except Exception:  # noqa: BLE001
            s2_epochs = max(int(epochs or 30), 300)
        log("检测到已有 S2 检查点 → 跳过预处理与 S1，直接续训 S2（总 epochs=%d）" % s2_epochs)
        set_state(step="7/8 续训 S2（语义→语音，总 epochs=%d）" % s2_epochs)
        os.makedirs(s2_ckpt_dir, exist_ok=True)
        os.makedirs(os.path.join(WORK_ROOT, "s2_weights", name), exist_ok=True)
        # 重新生成 S2 配置（应用最新的保存频率等参数），s2_train.py 检测到检查点会自动 resume
        _make_s2_config(os.path.join(work, "s2_%s.json" % name), opt_dir, name, s2_epochs, batch=_auto_batches()[1])
        run_proc([PY312, os.path.join(GSV_ROOT, "GPT_SoVITS", "s2_train.py"),
                  "-c", os.path.join(work, "s2_%s.json" % name)],
                 GSV_ROOT, "S2 续训", pythonpath=GSV_PYTHONPATH)
        set_state(step="8/8 发布模型")
        _publish_gpt(name, work)
        _clean_work(name, "gpt")
        return
    if os.path.isdir(work):
        shutil.rmtree(work, ignore_errors=True)
    # 清掉该角色旧的训练产物，避免残留检查点导致恢复失败（torch 2.6+ weights_only 报错）或误用旧权重
    for sub in ("s1_logs", "s2_logs", "s1_half_weights", "s2_weights"):
        p = os.path.join(WORK_ROOT, sub, name)
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    sliced = os.path.join(work, "sliced")
    os.makedirs(sliced, exist_ok=True)

    set_state(step="1/8 切分音频")
    try:
        run_proc([PY312, os.path.join(GSV_ROOT, "tools", "slice_audio.py"),
                  raw, sliced, "-35", "4000", "300", "10", "500", "0.9", "0.25", "0", "1"],
                 GSV_ROOT, "音频切分", pythonpath=GSV_PYTHONPATH)
    except Exception:
        log("切分脚本不可用，直接用原始音频（长音频效果会差）")
        for a in audios:
            shutil.copy(a, os.path.join(sliced, os.path.basename(a)))
    wavs = glob_wavs(sliced)
    if not wavs:
        raise RuntimeError("没有可用的切片音频")
    log("切片音频：%d 个" % len(wavs))
    # 修复：连续说话无停顿的素材切片脚本切不碎（可能切出 60~96 秒超长段），
    # 超长切片会让训练特征巨大、计算爆炸卡死；这里把 >12 秒的二次切成 8 秒段
    _reslice_long(sliced)
    wavs = glob_wavs(sliced)
    if not wavs:
        raise RuntimeError("切片后没有可用音频")

    opt_dir = os.path.join(work, "fine_tune_dataset", name)
    os.makedirs(opt_dir, exist_ok=True)

    set_state(step="2/8 语音识别标字（SenseVoice）")
    inp_text = os.path.join(work, "input_list.txt")
    n_ok = _asr_slices(wavs, inp_text)
    if n_ok == 0:
        raise RuntimeError("语音识别全部失败，没有可用的文字标注（请确认素材是清晰中文人声）")
    log("语音识别成功 %d/%d 段" % (n_ok, len(wavs)))

    # ---- 标字校对环节：方言/需要校对的素材停下等页面核对；普通话默认自动跳过（全自动队列）----
    _wait_review(name, wavs, inp_text, review=review)

    base_env = {
        "inp_text": inp_text,
        "inp_wav_dir": sliced,
        "exp_name": name,
        "i_part": "0",
        "all_parts": "1",
        "opt_dir": opt_dir,
        "bert_pretrained_dir": os.path.join(GSV_ROOT, "GPT_SoVITS", "pretrained_models", "chinese-roberta-wwm-ext-large"),
        "cnhubert_base_dir": os.path.join(GSV_ROOT, "GPT_SoVITS", "pretrained_models", "chinese-hubert-base"),
        "pretrained_s2G": GPT_BASE["s2G"],
        "s2config_path": os.path.join(GSV_ROOT, "GPT_SoVITS", "configs", "s2.json"),
        "is_half": "True",
    }
    for miss in [k for k in ("s1bert", "s2G", "s2D") if not os.path.isfile(GPT_BASE[k])]:
        raise RuntimeError("缺少 GPT-SoVITS 基座模型 %s，请先在环境检查页下载" % miss)

    set_state(step="3/8 提取文本/音素/BERT 特征")
    run_proc_env([PY312, os.path.join(GSV_ROOT, "GPT_SoVITS", "prepare_datasets", "1-get-text.py")],
                 GSV_ROOT, base_env, "文本音素提取", pythonpath=GSV_PYTHONPATH)
    _merge_gpt_parts(opt_dir)
    if not os.path.isfile(os.path.join(opt_dir, "2-name2text.txt")):
        raise RuntimeError("标字/音素提取失败：未生成 2-name2text.txt")

    set_state(step="4/8 提取 HuBERT 特征")
    run_proc_env([PY312, os.path.join(GSV_ROOT, "GPT_SoVITS", "prepare_datasets", "2-get-hubert-wav32k.py")],
                 GSV_ROOT, base_env, "HuBERT 特征", pythonpath=GSV_PYTHONPATH)
    if not (os.path.isdir(os.path.join(opt_dir, "4-cnhubert"))
            and os.path.isdir(os.path.join(opt_dir, "5-wav32k"))):
        raise RuntimeError("HuBERT 特征提取失败：未生成 4-cnhubert/5-wav32k")

    set_state(step="5/8 提取语义")
    run_proc_env([PY312, os.path.join(GSV_ROOT, "GPT_SoVITS", "prepare_datasets", "3-get-semantic.py")],
                 GSV_ROOT, base_env, "语义提取", pythonpath=GSV_PYTHONPATH)
    _merge_gpt_parts(opt_dir)
    if not os.path.isfile(os.path.join(opt_dir, "6-name2semantic.tsv")):
        raise RuntimeError("语义提取失败：未生成 6-name2semantic.tsv")

    # 训练量自动规划：S1 默认 8000 步（素材 10 分钟左右合理量，TRAIN_TARGET_STEPS 可覆盖）；
    # S2 默认 4000 步（可 TRAIN_S2_TARGET_STEPS 覆盖）——S2 每轮有两次大权重落盘，步数太多耗时很长
    total_sec = sum(_audio_dur(a) for a in audios)
    target_steps = _target_steps_by_data(total_sec)
    s2_target = int(os.environ.get("TRAIN_S2_TARGET_STEPS", "4000"))
    # 按显卡显存自动选 batch（8G 低配 / 8-12G 中配 / 12G+ 高配），不同电脑用不同方案
    s1_batch, s2_batch = _auto_batches()
    # 训练集 loader 会把不足 100 条的切片自动复制到 100 条，轮数必须按有效条数=max(100, len(wavs)) 计算，
    # 否则少素材角色会多训 4~6 倍步数（大彬案例：3048 轮≈1.8 万步，跑 12 小时）
    eff_n = max(100, len(wavs))
    s1_epochs = max(int(epochs or 30), (target_steps * s1_batch + eff_n - 1) // eff_n)
    s2_epochs = max(int(epochs or 30), (s2_target * s2_batch + eff_n - 1) // eff_n)
    log("GPT 训练计划：切片 %d 个，素材 %.1f 分钟 → S1 %d 步 epochs=%d(batch %d)，S2 %d 步 epochs=%d(batch %d)" % (
        len(wavs), total_sec / 60, target_steps, s1_epochs, s1_batch, s2_target, s2_epochs, s2_batch))

    set_state(step="6/8 训练 S1（文本→语义，epochs=%d）" % s1_epochs)
    s1_cfg = os.path.join(work, "s1_%s.yaml" % name)
    _make_s1_config(s1_cfg, opt_dir, name, s1_epochs, batch=s1_batch)
    # S1 推理格式权重保存到 s1_half_weights/<名>，必须先建目录（process_ckpt.my_save 不会自动建）
    os.makedirs(os.path.join(WORK_ROOT, "s1_half_weights", name), exist_ok=True)
    run_proc([PY312, os.path.join(GSV_ROOT, "GPT_SoVITS", "s1_train.py"),
              "-c", s1_cfg], GSV_ROOT, "S1 训练", pythonpath=GSV_PYTHONPATH)

    set_state(step="7/8 训练 S2（语义→语音，epochs=%d）" % s2_epochs)
    s2_cfg = os.path.join(work, "s2_%s.json" % name)
    _make_s2_config(s2_cfg, opt_dir, name, s2_epochs, batch=s2_batch)
    # S2 保存权重到 <opt_dir>/logs_s2_<version>，必须先建目录（utils.my_save 不会自动建）
    os.makedirs(os.path.join(opt_dir, "logs_s2_v2"), exist_ok=True)
    # S2 推理格式权重保存到 s2_weights/<名>，同样要先建目录
    os.makedirs(os.path.join(WORK_ROOT, "s2_weights", name), exist_ok=True)
    run_proc([PY312, os.path.join(GSV_ROOT, "GPT_SoVITS", "s2_train.py"),
              "-c", s2_cfg], GSV_ROOT, "S2 训练", pythonpath=GSV_PYTHONPATH)

    set_state(step="8/8 发布模型")
    _publish_gpt(name, work)
    _clean_work(name, "gpt")


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def _asr_slices(wavs, out_path):
    """SenseVoice 把切片语音转成文字，写入 GPT-SoVITS 需要的 input_list（path|spk|lang|text）。"""
    try:
        from funasr import AutoModel
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("ASR 依赖 funasr 不可用：%s" % exc)
    asr_dir = os.path.join(PROJECT_ROOT, "gptsovits", "asr", "SenseVoiceSmall")
    device = "cuda" if _cuda_available() else "cpu"
    model = AutoModel(model=asr_dir, trust_remote_code=False, device=device, disable_update=True)
    lines = []
    for w in wavs:
        try:
            res = model.generate(input=w)[0]
            text = re.sub(r"<\|[^|]*\|>", "", res.get("text", "")).strip()
            if text:
                lines.append("%s|0|zh|%s" % (w, text))
        except Exception as exc:  # noqa: BLE001
            log("ASR 失败 %s: %s" % (os.path.basename(w), str(exc)[:120]))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def _merge_gpt_parts(opt_dir):
    """prepare 脚本输出带 -0 后缀的文件，复制成训练脚本需要的固定文件名。"""
    pairs = [
        ("2-name2text-0.txt", "2-name2text.txt"),
        ("6-name2semantic-0.tsv", "6-name2semantic.tsv"),
    ]
    for src, dst in pairs:
        sp = os.path.join(opt_dir, src)
        if os.path.isfile(sp) and not os.path.isfile(os.path.join(opt_dir, dst)):
            shutil.copy(sp, os.path.join(opt_dir, dst))
            log("数据集文件：%s → %s" % (src, dst))


def _wait_review(name, wavs, inp_text, review=True):
    """标字校对环节。review=True 时暂停训练等页面核对/修改；review=False 时自动跳过（普通话全自动队列）。"""
    if not review:
        log("自动跳过标字校对（未勾选“需要校对”，默认按原始识别文字继续）")
        return
    global _review_pending
    items = []
    with open(inp_text, encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]
    for w in wavs:
        base = os.path.basename(w)
        text = ""
        for l in lines:
            parts = l.split("|")
            if len(parts) >= 4 and os.path.basename(parts[0]) == base:
                text = parts[3].strip()
                break
        items.append({"file": base, "text": text})
    _review_pending = {"name": name, "wavs": wavs, "items": items, "inp_text": inp_text}
    _review_evt.clear()
    set_state(step="标字校对中（%d 段，请核对识别文字；普通话可跳过，方言务必逐句校对）" % len(items))
    log("▶ 标字校对：请在页面核对/修改识别文字后点“确认继续”；普通话识别准确可点“跳过校对”")
    _review_evt.wait()
    _review_pending = {}


def _gpu_mem_gb():
    """检测 GPU 总显存（GB）；无 GPU/失败返回 0。"""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        pass
    return 0


def _auto_batches():
    """按显卡显存自动选 batch 档位（不同电脑用不同方案）：
    - 显存 ≤ 8GB（低配）：S1 batch 4 / S2 batch 2
    - 显存 8~12GB（中配）：S1 batch 6 / S2 batch 3
    - 显存 ≥ 12GB（高配）：S1 batch 8 / S2 batch 4
    环境变量 TRAIN_S1_BATCH / TRAIN_S2_BATCH 可强制覆盖。返回 (s1_batch, s2_batch)。"""
    s1 = int(os.environ.get("TRAIN_S1_BATCH", "0") or 0)
    s2 = int(os.environ.get("TRAIN_S2_BATCH", "0") or 0)
    if s1 > 0 and s2 > 0:
        return s1, s2
    mem = _gpu_mem_gb()
    if mem and mem <= 8:
        return (s1 or 4), (s2 or 2)
    if mem and mem <= 12:
        return (s1 or 6), (s2 or 3)
    return (s1 or 8), (s2 or 4)


def run_proc_env(cmd, cwd, extra_env, step, pythonpath=None):
    global _active_proc
    env = os.environ.copy()
    env.update(extra_env)
    env["PYTHONIOENCODING"] = "utf-8"
    if pythonpath:
        env["PYTHONPATH"] = pythonpath + os.pathsep + env.get("PYTHONPATH", "")
    if pythonpath and cmd and str(cmd[0]) == PY312 and str(cmd[1]).lower().endswith(".py"):
        cmd = [PY312, "-c", WRAPPER, pythonpath, str(cmd[1])] + [str(x) for x in cmd[2:]]
    log("▶ %s" % step)
    if _stop_requested:
        raise RuntimeError("训练已手动停止")
    p = subprocess.Popen(
        [str(c) for c in cmd], cwd=cwd, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, env=env, bufsize=1, text=True,
        encoding="utf-8", errors="replace")
    _active_proc = p
    try:
        for line in p.stdout:
            line = line.rstrip("\n\r")
            if line:
                log(line[:300])
        p.wait()
    finally:
        _active_proc = None
    if p.returncode != 0:
        raise RuntimeError("%s 失败（exit=%s）" % (step, p.returncode))


def _make_s1_config(path, opt_dir, name, epochs, batch=8):
    """生成完整的 S1 训练配置（仓库自带 yaml 缺训练必需字段，这里从完整模板生成）。
    batch 由 _auto_batches 按显卡显存自动选择。"""
    import yaml
    cfg = {
        "train": {
            "seed": 1234,
            "epochs": int(epochs),
            "batch_size": int(batch),
            # 保存频率：每 max(1, epochs//25) epoch 保存一次检查点（保存时 Lightning 先在内存序列化 0.9GB+，
            # 频率太高容易触发 MemoryError；本机 32GB 内存实测每 14 epoch 保存会失败）
            "save_every_n_epoch": max(1, int(epochs) // 25),
            "if_save_latest": True,
            "if_save_every_weights": True,  # 同时保存推理格式权重（<名>-e<轮>.ckpt，api 需要 weight+config）
            "half_weights_save_dir": os.path.join(WORK_ROOT, "s1_half_weights", name).replace("\\", "/"),
            "exp_name": name,
            "precision": "16-mixed",
            "gradient_clip": 1.0,
        },
        "optimizer": {
            "lr": 0.01,
            "lr_init": 0.00001,
            "lr_end": 0.0001,
            "warmup_steps": 2000,
            "decay_steps": 40000,
        },
        "data": {
            "max_eval_sample": 8,
            "max_sec": 54,
            # 数据加载线程 1 + 预取 2（data_module）：Windows 上 worker 复制内存 + 预取 16 会让
            # S1 内存暴涨到 17GB 拖慢训练；=0 又会卡在首个 epoch。1 个 worker + 小预取兼顾速度与内存
            "num_workers": 1,
            "pad_val": 1024,
        },
        "model": {
            "vocab_size": 1025,
            "phoneme_vocab_size": 732,
            "embedding_dim": 512,
            "hidden_dim": 512,
            "head": 16,
            "linear_units": 2048,
            "n_layer": 24,
            "dropout": 0,
            "EOS": 1024,
            "random_bert": 0,
        },
        "inference": {"top_k": 15},
        "output_dir": os.path.join(WORK_ROOT, "s1_logs", name).replace("\\", "/"),
        # 关键：从基座模型继续训练，否则从零训练质量差、声音不像
        "pretrained_s1": GPT_BASE["s1bert"].replace("\\", "/"),
        "train_phoneme_path": os.path.join(opt_dir, "2-name2text.txt").replace("\\", "/"),
        "train_semantic_path": os.path.join(opt_dir, "6-name2semantic.tsv").replace("\\", "/"),
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def _make_s2_config(path, opt_dir, name, epochs, batch=4):
    """生成完整的 S2 训练配置（仓库自带 s2.json 缺 gpu_numbers/exp_dir/save_every_epoch 等字段）。
    batch 由 _auto_batches 按显卡显存自动选择（16G 卡 4，8G 卡 2，避免显存占满崩溃）。"""
    s2_root = os.path.join(WORK_ROOT, "s2_logs", name).replace("\\", "/")
    cfg = {
        "train": {
            "log_interval": 100,
            "eval_interval": 500,
            "seed": 1234,
            "epochs": int(epochs),
            "learning_rate": 0.0001,
            "betas": [0.8, 0.99],
            "eps": 1e-09,
            "batch_size": int(batch),
            "fp16_run": True,
            "lr_decay": 0.999875,
            "segment_size": 20480,
            "init_lr_ratio": 1,
            "warmup_epochs": 0,
            "c_mel": 45,
            "c_kl": 1.0,
            "text_low_lr_rate": 0.4,
            "grad_ckpt": False,
            "gpu_numbers": "0",
            # 每轮都落盘 G+D 两次大权重（各 500MB+），磁盘慢时耗时明显；且保存时 torch.save 序列化
            # 会有 3-4GB 内存峰值，本机 32GB 内存训练时高频保存易崩。按总轮数每 15 次保存一次
            "save_every_epoch": max(1, int(epochs) // 15),
            "if_save_latest": 1,
            "if_save_every_weights": True,  # 同时保存推理格式权重（weight+config，api 直接可用）
            "pretrained_s2G": GPT_BASE["s2G"],
            "pretrained_s2D": GPT_BASE["s2D"],
        },
        "save_weight_dir": os.path.join(WORK_ROOT, "s2_weights", name).replace("\\", "/"),
        "data": {
            "max_wav_value": 32768.0,
            "sampling_rate": 32000,
            "filter_length": 2048,
            "hop_length": 640,
            "win_length": 2048,
            "n_mel_channels": 128,
            "mel_fmin": 0.0,
            "mel_fmax": None,
            "add_blank": True,
            "n_speakers": 300,
            "cleaned_text": True,
            # TextAudioSpeakerLoader 直接用 exp_dir 找 2-name2text.txt / 4-cnhubert / 5-wav32k
            "exp_dir": opt_dir.replace("\\", "/"),
            "training_files": os.path.join(opt_dir, "5-name2wav32k.tsv").replace("\\", "/"),
            "val_files": os.path.join(opt_dir, "5-name2wav32k.tsv").replace("\\", "/"),
            "semantic_files": os.path.join(opt_dir, "6-name2semantic.tsv").replace("\\", "/"),
            "hubert_files": os.path.join(opt_dir, "4-cnhubert.tsv").replace("\\", "/"),
        },
        "model": {
            "inter_channels": 192,
            "hidden_channels": 192,
            "filter_channels": 768,
            "n_heads": 2,
            "n_layers": 6,
            "kernel_size": 3,
            "p_dropout": 0.1,
            "resblock": "1",
            # 与基座 s2G2333k.pth 一致（仓库自带 s2.json 写成了 [3,5,11]，加载基座会尺寸报错）
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_rates": [10, 8, 2, 2, 2],
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": [16, 16, 8, 2, 2],
            "n_layers_q": 3,
            "use_spectral_norm": False,
            "gin_channels": 512,
            "semantic_frame_rate": "25hz",
            "freeze_quantizer": True,
            "version": "v2",
        },
        "s2_ckpt_dir": s2_root,
        "content_module": "cnhubert",
        "name": name,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def _latest_recursive(folder, pattern):
    import glob
    hits = glob.glob(os.path.join(folder, "**", pattern), recursive=True)
    if not hits:
        return None
    return max(hits, key=os.path.getmtime)


def _publish_gpt(name, work):
    # 交付：GPT-SoVITS 模型只生成交付包（交付模型/<角色>/ 4件套），不自动推送任何项目；
    # 用户把该文件夹复制到 文字驱动/对话模型 的 tts_service\models 即被自动识别（目录扫描）
    out_dir = os.path.join(DELIVERY_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    # 优先取推理格式权重（s1_half_weights/<名>-e<轮>.ckpt），Lightning 检查点格式不能直接推理
    s1 = (_latest_recursive(os.path.join(WORK_ROOT, "s1_half_weights", name), "*.ckpt")
          or _latest_recursive(os.path.join(WORK_ROOT, "s1_logs", name), "*.ckpt"))
    # 优先取推理格式权重（s2_weights/<名>_e*_s*.pth），训练检查点（model/optimizer）不能直接推理
    s2 = (_latest_recursive(os.path.join(WORK_ROOT, "s2_weights", name), "*.pth")
          or _latest_recursive(os.path.join(WORK_ROOT, "s2_logs", name), "G_*.pth")
          or _latest_recursive(os.path.join(work, "fine_tune_dataset", name), "G_*.pth"))
    if not s1 or not s2:
        raise RuntimeError("训练完成但未找到 s1/s2 权重（%s / %s）" % (s1, s2))
    shutil.copy(s1, os.path.join(out_dir, name + ".ckpt"))
    shutil.copy(s2, os.path.join(out_dir, name + ".pth"))
    _make_gpt_ref(work, out_dir, name)
    _ensure_gpt_ref_ok(out_dir, name)
    # output 存档（保留一份便于回看）
    pub_dir = os.path.join(OUTPUT_ROOT, name)
    os.makedirs(pub_dir, exist_ok=True)
    shutil.copy(s1, os.path.join(pub_dir, name + ".ckpt"))
    shutil.copy(s2, os.path.join(pub_dir, name + ".pth"))
    log("模型已交付：%s → 交付模型\\%s\\（4件套齐全），把该文件夹复制到 文字驱动/对话模型 的 tts_service\\models 即可使用" % (name, name))


def _ensure_gpt_ref_ok(out_dir, name):
    """发布前兜底校验 ref.wav：超过 10 秒强制裁剪到 8 秒并截断 ref_text.txt。
    超长/错乱参考是文字驱动合成漏字的主因（曾实测发布过 155 秒 ref + 300 字错乱文本的案例）。"""
    import librosa
    import soundfile as sf
    ref = os.path.join(out_dir, "ref.wav")
    if not os.path.isfile(ref):
        log("警告：%s 缺少 ref.wav，文字驱动项目将不可用该角色" % name)
        return
    try:
        dur = librosa.get_duration(path=ref)
    except Exception as exc:  # noqa: BLE001
        log("%s ref.wav 校验失败：%s" % (name, exc))
        return
    if dur > 10:
        y, sr = librosa.load(ref, sr=32000, mono=True)
        y = y[:8 * sr]
        sf.write(ref, y, 32000)
        rt = os.path.join(out_dir, "ref_text.txt")
        text = ""
        if os.path.isfile(rt):
            text = open(rt, encoding="utf-8").read().strip()
        with open(rt, "w", encoding="utf-8") as f:
            f.write(text[:40])
        log("⚠ %s 的 ref.wav 超长（%.1f 秒），已强制裁剪到 8 秒并截断 ref_text（防合成漏字）" % (name, dur))
    else:
        log("%s 参考音频校验通过（%.1f 秒）" % (name, dur))


def _make_gpt_ref(work, out_dir, name):
    """从训练切片里挑一段有代表性的清晰人声做参考音频（ref.wav + ref_text.txt）。
    强制保证 ref.wav 3~10 秒：超长参考（如 155 秒）会导致合成异常短/漏字/音色跑偏
    （雷军/大彬/冯彦林实测：文字驱动项目 8060 合成漏字多与 ref 超长+文本错乱有关）。"""
    import glob
    import librosa
    import soundfile as sf

    opt_dir = os.path.join(work, "fine_tune_dataset", name)
    txt_map = {}
    for fn in sorted(glob.glob(os.path.join(opt_dir, "2-name2text-*.txt"))):
        with open(fn, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4 and parts[0]:
                    txt_map[os.path.basename(parts[0])] = parts[3].strip()
    sliced = os.path.join(work, "sliced")
    candidates = []
    for wav in glob_wavs(sliced):
        text = txt_map.get(os.path.basename(wav), "").strip()
        if len(text) < 5:
            continue
        try:
            dur = librosa.get_duration(path=wav)
        except Exception:  # noqa: BLE001
            dur = 0
        # 优先时长合适（3~10 秒）的切片；时长合适优先于文本长短，避免选到超长切片
        if 3.0 <= dur <= 10.0:
            candidates.append((0, dur, wav, text))
    if not candidates:
        for wav in glob_wavs(sliced):
            text = txt_map.get(os.path.basename(wav), "").strip()
            if text:
                candidates.append((1, 0, wav, text))
    if not candidates:
        log("警告：没有可用切片生成 ref，请在 %s 手工放置 ref.wav/ref_text.txt" % out_dir)
        return
    candidates.sort(key=lambda x: (x[0], x[1]))
    _, dur, src_wav, text = candidates[0]
    y, sr = librosa.load(src_wav, sr=32000, mono=True)
    # 参考音频必须 3~10 秒：超长参考会导致合成结果异常短/音色跑偏（雷军/大彬实测）。
    # ASR 转写成功则用转写文本；ASR 失败也【必须】裁剪音频，绝不发布超长 ref。
    if dur > 10:
        seg = y[:8 * sr]
        new_text = text[:40]  # 兜底：按原标注文本截断（8 秒音频约对应 40 字内）
        try:
            from funasr import AutoModel
            asr_dir = os.path.join(PROJECT_ROOT, "gptsovits", "asr", "SenseVoiceSmall")
            device = "cuda" if _cuda_available() else "cpu"
            asr = AutoModel(model=asr_dir, trust_remote_code=False, device=device, disable_update=True)
            tmp = os.path.join(work, "_ref_tmp.wav")
            sf.write(tmp, seg, 32000)
            res = asr.generate(input=tmp)[0]
            new_text = re.sub(r"<\|[^|]*\|>", "", res.get("text", "")).strip()
            os.remove(tmp)
            if not new_text:
                new_text = text[:40]
        except Exception as exc:  # noqa: BLE001
            log("参考音频超长裁剪 ASR 失败，音频仍强制裁剪到 8 秒：%s" % exc)
            new_text = text[:40]
        y = seg
        text = new_text or text[:40]
        dur = 8.0
    # 最终兜底：无论如何 ref.wav 不超过 10 秒（防止手工放置/其他来源的超长 ref）
    if y.shape[0] > 10 * sr:
        y = y[:8 * sr]
        text = (text or "")[:40]
        dur = 8.0
    sf.write(os.path.join(out_dir, "ref.wav"), y, 32000)
    with open(os.path.join(out_dir, "ref_text.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    log("已生成参考音频 ref.wav（%s，%.1f 秒）与 ref_text.txt：%s" % (
        os.path.basename(src_wav), dur, text[:40]))


def _clean_work(name, engine):
    """发布成功后清理该角色的全部训练中间产物（训练中心只负责训练+交付，不留垃圾，方便下次训练别的角色）。

    - rvc：换声项目 rvc\\logs\\<角色>（RVC 训练中间产物）
    - gpt：work 下该角色的全部中间产物（切片/特征/检查点等）
    模型已生成交付包（交付模型） + output 存档，中间产物不再需要。
    """
    removed = []
    if engine == "rvc":
        exp = os.path.join(RVC_ROOT, "logs", name)
        if os.path.isdir(exp):
            shutil.rmtree(exp, ignore_errors=True)
            removed.append(exp)
    elif engine == "gpt":
        for p in (os.path.join(WORK_ROOT, "gpt_" + name),
                  os.path.join(WORK_ROOT, "prep_" + name),
                  os.path.join(WORK_ROOT, "uploaded", name),
                  os.path.join(WORK_ROOT, "single_" + name)):
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
                removed.append(p)
        for sub in ("s1_logs", "s2_logs", "s1_half_weights", "s2_weights"):
            p = os.path.join(WORK_ROOT, sub, name)
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
                removed.append(p)
    for p in removed:
        log("已清理中间产物：%s" % p)
    if removed:
        log("清理完成：%s 训练中间产物已全部清除（模型已交付，未占本机空间）" % name)
    else:
        log("无残留中间产物可清理")


# ---------------- 下载基座模型 ----------------
def download_file(url, dst, label):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    log("下载 %s → %s" % (label, dst))
    tmp = dst + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                log("  %s: %.1f%%" % (label, 100.0 * done / total))
    shutil.move(tmp, dst)
    log("完成 %s" % label)


def download_base(engine):
    if engine == "gpt":
        for k, url in GPT_DL.items():
            if not os.path.isfile(GPT_BASE[k]):
                download_file(url, GPT_BASE[k], k)
    elif engine == "rvc":
        for k, url in RVC_DL.items():
            if not os.path.isfile(RVC_BASE[k]):
                try:
                    download_file(url, RVC_BASE[k], "f0" + k + "48k")
                except Exception as exc:
                    log("RVC 基座模型下载失败（%s）：%s" % (k, exc))
                    log("请手动下载 f0G48k.pth / f0D48k.pth 放到 %s" % os.path.dirname(RVC_BASE["G"]))
    else:
        raise RuntimeError("未知引擎: %s" % engine)


# ---------------- FastAPI ----------------
app = FastAPI(title="训练中心")

# 页面模板独立维护在 index_template.html（整体优化 + 任务队列 UI）
_TPL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index_template.html")
INDEX_HTML = open(_TPL, encoding="utf-8").read()

@app.get("/", response_class=HTMLResponse)
def index():
    if TRAIN_MODE == "huan_sheng":
        engine_html = ('<input type="hidden" id="engine" value="rvc">'
                       '<b style="color:#6d28d9">RVC 换音色（内容/时长保留）→ 生成交付模型（交付模型\\rvc\\<角色>\\）</b>')
        base_btn = "下载 RVC 基座模型"
    else:
        engine_html = ('<input type="hidden" id="engine" value="gpt">'
                       '<b style="color:#6d28d9">GPT-SoVITS 文字驱动（TTS）→ 生成交付模型（交付模型\\<角色>\\，4件套）</b>')
        base_btn = "下载 GPT-SoVITS 基座模型"
    return (INDEX_HTML
            .replace("__TITLE__", MODE_NAME)
            .replace("__DESC__", MODE_DESC)
            .replace("__ENGINE_HTML__", engine_html)
            .replace("__BASE_BTN__", base_btn))


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "train-center", "port": PORT,
            "mode": TRAIN_MODE, "mode_name": MODE_NAME, "engine": MODE_ENGINE}


@app.get("/api/status")
def status():
    with _lock:
        return dict(_state)


@app.get("/api/env")
def env_check():
    torch_ok = cuda_ok = False
    try:
        import torch
        torch_ok = True
        cuda_ok = torch.cuda.is_available()
    except Exception:
        pass
    datasets = [d for d in os.listdir(DATA_ROOT)
                if os.path.isdir(os.path.join(DATA_ROOT, d))]
    return {
        "python": PY312,
        "运行模式": MODE_NAME,
        "训练引擎": MODE_ENGINE,
        "交付目录": (RVC_DELIVERY_DIR if TRAIN_MODE == "huan_sheng" else DELIVERY_ROOT),
        "gpu": cuda_ok,
        "gpu_mem_gb": round(_gpu_mem_gb(), 1),
        "s1_batch": _auto_batches()[0],
        "s2_batch": _auto_batches()[1],
        "rvc_engine": os.path.isdir(RVC_ROOT),
        "gpt_engine": os.path.isdir(GSV_ROOT),
        "rvc_base_G": os.path.isfile(RVC_BASE["G"]),
        "rvc_base_D": os.path.isfile(RVC_BASE["D"]),
        "gpt_base_s1bert": os.path.isfile(GPT_BASE["s1bert"]),
        "gpt_base_s2G": os.path.isfile(GPT_BASE["s2G"]),
        "gpt_base_s2D": os.path.isfile(GPT_BASE["s2D"]),
        "数据集目录": DATA_ROOT,
        "已有数据集": datasets,
        "提示": "基座模型缺失时可点页面上的下载按钮；RVC 基座若镜像下载失败，需手动放置 f0G48k.pth/f0D48k.pth",
    }


@app.post("/api/train")
def train(engine: str = Form("rvc"), name: str = Form(""), epochs: int = Form(30), clean: bool = Form(True)):
    # 解耦：忽略传入引擎，只允许当前模式的引擎（换声模式只训 RVC，文字驱动模式只训 GPT-SoVITS）
    return _start_job(MODE_ENGINE, name, epochs, clean)


@app.post("/api/train_json")
def train_json(req: dict):
    name = (req.get("name") or "").strip()
    epochs = int(req.get("epochs") or 30)
    clean = bool(req.get("clean", True))
    review = bool(req.get("review", False))
    return _start_job(MODE_ENGINE, name, epochs, clean, review=review)


@app.post("/api/train_dir")
def train_dir(req: dict):
    """直接用服务器上的文件夹路径作为训练数据（无需上传）"""
    name = (req.get("name") or "").strip()
    epochs = int(req.get("epochs") or 30)
    clean = bool(req.get("clean", True))
    review = bool(req.get("review", False))
    folder = (req.get("dir") or req.get("folder") or "").strip()
    if not folder or not os.path.exists(folder):
        return JSONResponse({"detail": "路径不存在：%s" % folder}, status_code=400)
    # 支持直接给单个音频文件：自动包一层目录，其余照旧自动切分/清洗/训练
    if os.path.isfile(folder):
        single_dir = os.path.join(WORK_ROOT, "single_" + (name or "audio"))
        os.makedirs(single_dir, exist_ok=True)
        dst = os.path.join(single_dir, os.path.basename(folder))
        shutil.copy(folder, dst)
        folder = single_dir
    audios = list_audio(folder)
    if not audios:
        return JSONResponse({"detail": "该文件夹里没有音频文件（支持 wav/mp3/m4a/flac/aac/ogg）"}, status_code=400)
    return _start_job(MODE_ENGINE, name, epochs, clean, raw_dir=folder, review=review)


@app.post("/api/estimate")
def estimate_api(req: dict):
    """预估训练时长（选好文件夹后调用）"""
    folder = (req.get("dir") or "").strip()
    if not folder or not os.path.isdir(folder):
        return JSONResponse({"detail": "文件夹路径不存在"}, status_code=400)
    est = estimate_training(folder)
    if not est:
        return JSONResponse({"detail": "该文件夹里没有音频文件"}, status_code=400)
    return est


@app.post("/api/train_upload")
async def train_upload(
    engine: str = Form("rvc"),
    name: str = Form(""),
    epochs: int = Form(30),
    clean: bool = Form(True),
    queue: bool = Form(False),
    review: bool = Form(False),
    files: list[UploadFile] = File(...),
):
    """上传文件夹里的音频文件，保存到 work/uploaded/<name>/ 后训练。
    queue=True 时只保存不启动训练（前端随后加入任务队列），返回保存目录。"""
    name = (name or "").strip()
    audios = [f for f in files if os.path.splitext(f.filename or "")[1].lower() in
              (".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg")]
    if not audios:
        return JSONResponse({"detail": "没有有效的音频文件（支持 wav/mp3/m4a/flac/aac/ogg）"}, status_code=400)
    up_dir = os.path.join(WORK_ROOT, "uploaded", name)
    if os.path.isdir(up_dir):
        shutil.rmtree(up_dir, ignore_errors=True)
    os.makedirs(up_dir, exist_ok=True)
    saved = []
    for f in audios:
        raw_name = os.path.basename(f.filename or "audio.wav")
        dst = os.path.join(up_dir, raw_name)
        n = 1
        while os.path.exists(dst):
            stem, ext = os.path.splitext(raw_name)
            dst = os.path.join(up_dir, "%s_%d%s" % (stem, n, ext))
            n += 1
        with open(dst, "wb") as out:
            data = await f.read()
            out.write(data)
        saved.append(os.path.basename(dst))
    log("已接收上传音频 %d 个 -> %s" % (len(saved), up_dir))
    if queue:
        return {"message": "音频已保存（等待加入队列）", "dir": up_dir}
    return _start_job(MODE_ENGINE, name, epochs, clean, raw_dir=up_dir, review=review)


@app.post("/api/download_base")
def download_base_api(engine: str = Form("gpt")):
    return _start_job("download_" + MODE_ENGINE, "", 0)


@app.post("/api/download_base_json")
def download_base_json(req: dict):
    return _start_job("download_" + MODE_ENGINE, "", 0)


def _start_job(kind, name, epochs, clean=True, raw_dir=None, review=False):
    global _stop_requested
    _cancel_exit()
    with _lock:
        if _state["running"]:
            return JSONResponse({"detail": "已有任务在运行，请等待"}, status_code=400)
        if kind == "download_" + MODE_ENGINE:
            pass  # 下载当前模式的基座模型
        elif kind != MODE_ENGINE:
            return JSONResponse({"detail": "当前为「%s」模式，只支持训练 %s 模型（换声/文字驱动请用对应启动脚本）"
                                        % (MODE_NAME, MODE_ENGINE)}, status_code=400)
        if not kind.startswith("download_") and not re.fullmatch(r"[A-Za-z0-9_\u4e00-\u9fff]{1,40}", name or ""):
            return JSONResponse({"detail": "角色名只能是中文/英文/数字/下划线，1~40 字符"}, status_code=400)
        _stop_requested = False
        _state.update(running=True, ok=False, error="", log=[], engine=kind, name=name, step="启动")
    def worker():
        ok, err = _execute_task(kind, name, epochs, clean, raw_dir, review=review)
        if ok:
            set_state(ok=True, step="完成", running=False)
        else:
            if _stop_requested:
                log("⏹ 训练已手动停止")
                set_state(error="已手动停止", running=False, step="已停止")
            else:
                log("✘ 失败：%s" % err)
                set_state(error=err, running=False, step="失败")
        _after_task_done()
    threading.Thread(target=worker, daemon=True).start()
    return {"message": "任务已开始", "engine": kind, "name": name}


def _execute_task(kind, name, epochs, clean=True, raw_dir=None, review=False):
    """执行单个任务，返回 (是否成功, 错误信息)。手动停止时返回 (False, "已手动停止")。"""
    try:
        if kind == "rvc":
            train_rvc(name, epochs, clean=clean, raw_dir=raw_dir)
        elif kind == "gpt":
            train_gpt(name, epochs, clean=clean, raw_dir=raw_dir, review=review)
        elif kind == "download_rvc":
            download_base("rvc")
        elif kind == "download_gpt":
            download_base("gpt")
        else:
            raise RuntimeError("未知任务: %s" % kind)
        return True, ""
    except Exception as exc:  # noqa: BLE001
        if _stop_requested:
            return False, "已手动停止"
        traceback.print_exc()
        return False, str(exc)


def _after_task_done():
    """当前任务结束后调用：有队列任务则继续跑下一个，否则所有任务完成后自动退出。
    注意：schedule_exit 不能在本函数持锁时调用（内部也要拿 _queue_lock，会死锁）。"""
    global _stop_requested
    _cancel_exit()
    should_exit = False
    with _queue_lock:
        if not _queue:
            should_exit = True
        else:
            task = _queue.pop(0)
            task["status"] = "running"
            kind = task["engine"]
            name = task["name"]
            epochs = task["epochs"]
            clean = task.get("clean", True)
            raw_dir = task.get("raw_dir")
            review = task.get("review", False)
    if should_exit:
        schedule_exit()
        return
    _stop_requested = False
    _state.update(running=True, ok=False, error="", log=[], engine=kind, name=name, step="启动")
    log("▶▶▶ 队列任务开始：%s（剩余排队 %d 个）" % (name, len(_queue)))

    def qworker():
        ok, err = _execute_task(kind, name, epochs, clean, raw_dir, review=review)
        with _queue_lock:
            if ok:
                task["status"] = "done"
                task["result"] = "成功"
                set_state(ok=True, step="完成", running=False)
            else:
                if _stop_requested:
                    task["status"] = "stopped"
                    task["result"] = "已手动停止"
                    log("⏹ 队列任务已手动停止：%s" % name)
                    set_state(error="已手动停止", running=False, step="已停止")
                else:
                    task["status"] = "failed"
                    task["result"] = err
                    log("✘ 失败：%s" % err)
                    set_state(error=err, running=False, step="失败")
            _queue_done.append(task)
            while len(_queue_done) > 30:
                _queue_done.pop(0)
        _after_task_done()
    threading.Thread(target=qworker, daemon=True).start()


def _enqueue_tasks(task_list):
    """把任务列表加入队列（校验角色名）。返回 (成功数, 加入的角色名列表)。"""
    added = []
    with _queue_lock:
        for t in task_list:
            name = (t.get("name") or "").strip()
            if not re.fullmatch(r"[A-Za-z0-9_\u4e00-\u9fff]{1,40}", name or ""):
                continue
            epochs = int(t.get("epochs") or 30)
            raw_dir = (t.get("dir") or "").strip() or None
            review = bool(t.get("review", False))
            _queue.append({"engine": MODE_ENGINE, "name": name, "epochs": epochs,
                           "clean": True, "raw_dir": raw_dir, "review": review,
                           "status": "waiting", "result": ""})
            added.append(name)
    return len(added), added


@app.get("/api/review_status")
def review_status_api():
    """查询当前是否有待校对的标字任务（含每段识别文字）。"""
    if not _review_pending:
        return {"pending": False}
    return {"pending": True, "name": _review_pending["name"],
            "items": [dict(it) for it in _review_pending["items"]]}


@app.post("/api/review_submit")
def review_submit_api(req: dict):
    """提交校对后的文字（items=[{file, text}, ...]），用校对结果重写标注并继续训练。"""
    if not _review_pending:
        return JSONResponse({"detail": "当前没有待校对任务"}, status_code=400)
    items = req.get("items") or []
    if not items:
        return JSONResponse({"detail": "没有提交内容"}, status_code=400)
    text_by_file = {str(it.get("file", "")): str(it.get("text") or "").strip() for it in items}
    with open(_review_pending["inp_text"], "w", encoding="utf-8") as f:
        for w in _review_pending["wavs"]:
            text = text_by_file.get(os.path.basename(w), "")
            if text:
                f.write("%s|0|zh|%s\n" % (w, text))
    log("标字校对完成：%d 段已确认，继续训练" % len(_review_pending["wavs"]))
    _review_evt.set()
    return {"message": "校对完成，继续训练"}


@app.post("/api/review_skip")
def review_skip_api():
    """跳过校对，直接用原始识别文字继续训练。"""
    if not _review_pending:
        return JSONResponse({"detail": "当前没有待校对任务"}, status_code=400)
    log("跳过标字校对，使用原始识别文字继续训练")
    _review_evt.set()
    return {"message": "已跳过校对，继续训练"}


@app.post("/api/train_queue")
def train_queue_api(req: dict):
    """批量加入训练队列：tasks=[{name, epochs?, dir?}, ...]；加入后自动按顺序依次训练。
    已加入队列的角色名，前端可刷新 /api/queue_status 查看状态。"""
    tasks = req.get("tasks") or []
    if not tasks:
        return JSONResponse({"detail": "没有任务"}, status_code=400)
    n, added = _enqueue_tasks(tasks)
    if n == 0:
        return JSONResponse({"detail": "没有有效的任务（角色名只能是中文/英文/数字/下划线，1~40 字符）"}, status_code=400)
    with _lock:
        idle = not _state["running"]
    if idle:
        _after_task_done()  # 无任务在跑 → 立即启动队列（注意不能在持锁时调用）
    return {"message": "已加入队列 %d 个任务（依次自动训练）" % n, "added": added}


@app.post("/api/train_all_datasets")
def train_all_datasets_api(req: dict):
    """一键把数据集目录 ziliao\\xunlianshuju\\ 下所有角色加入队列，依次自动训练。"""
    dirs = sorted(d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d)))
    if not dirs:
        return JSONResponse({"detail": "数据集目录 %s 为空" % DATA_ROOT}, status_code=400)
    tasks = [{"name": d, "epochs": int(req.get("epochs") or 30),
              "review": bool(req.get("review", False))} for d in dirs]
    n, added = _enqueue_tasks(tasks)
    with _lock:
        idle = not _state["running"]
    if idle:
        _after_task_done()
    return {"message": "已将 %d 个数据集加入队列（依次自动训练）：%s" % (n, "、".join(added)), "added": added}


@app.get("/api/queue_status")
def queue_status_api():
    """查询任务队列状态：当前任务、等待队列、已完成。"""
    with _queue_lock:
        waiting = [dict(t) for t in _queue]
        done = [dict(t) for t in _queue_done]
    with _lock:
        current = None
        if _state["running"]:
            current = {"engine": _state["engine"], "name": _state["name"], "step": _state["step"]}
    return {"running": _state["running"], "current": current,
            "waiting": waiting, "done": done}


@app.post("/api/queue_stop")
def queue_stop_api():
    """停止当前正在训练的任务：杀掉训练子进程树 + 标记手动停止。
    任务被标记为"已停止"；队列自动继续下一个任务（或全部完成自动退出）。"""
    global _stop_requested
    with _lock:
        if not _state["running"]:
            return JSONResponse({"detail": "当前没有正在训练的任务"}, status_code=400)
        _stop_requested = True
    # 若处于标字校对暂停阶段，放行事件让任务走到下一个训练步骤时被停止标志拦下
    if _review_pending:
        _review_evt.set()
    p = _active_proc
    if p and p.poll() is None:
        try:
            subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"],
                           capture_output=True, timeout=10)
        except Exception:  # noqa: BLE001
            try:
                p.kill()
            except Exception:  # noqa: BLE001
                pass
    log("⏹ 已请求停止当前训练任务（杀训练进程）")
    return {"message": "正在停止当前训练任务…"}


@app.post("/api/queue_clear")
def queue_clear_api():
    """清空等待队列（不中断正在训练的任务）。"""
    with _queue_lock:
        n = len(_queue)
        _queue.clear()
    return {"message": "已清空 %d 个等待任务（当前训练不受影响）" % n}


@app.post("/api/queue_remove")
def queue_remove_api(req: dict):
    """从等待队列删除指定角色任务；等待队列没有时，尝试从已完成/已停止列表删除。"""
    name = (req.get("name") or "").strip()
    with _queue_lock:
        for i, t in enumerate(_queue):
            if t["name"] == name:
                _queue.pop(i)
                return {"message": "已从等待队列移除：%s" % name, "removed": name}
        for i, t in enumerate(_queue_done):
            if t["name"] == name:
                _queue_done.pop(i)
                return {"message": "已从完成列表移除：%s" % name, "removed": name}
    return JSONResponse({"detail": "队列中没有 %s（可能正在训练）" % name}, status_code=400)


@app.post("/api/queue_move")
def queue_move_api(req: dict):
    """调整等待队列顺序：把 name 移到指定位置（index=0 表示下一个就训它）。"""
    name = (req.get("name") or "").strip()
    index = int(req.get("index") or 0)
    with _queue_lock:
        idx = next((i for i, t in enumerate(_queue) if t["name"] == name), None)
        if idx is None:
            return JSONResponse({"detail": "等待队列中没有 %s（可能正在训练或已完成）" % name}, status_code=400)
        task = _queue.pop(idx)
        index = max(0, min(index, len(_queue)))
        _queue.insert(index, task)
    return {"message": "已调整 %s 到队列第 %d 位（前面还有 %d 个）" % (name, index + 1, index)}


if __name__ == "__main__":
    logger.info("训练中心启动 port=%s auto_exit=%s", PORT, AUTO_EXIT)
    _server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=PORT, workers=1))
    _server.run()