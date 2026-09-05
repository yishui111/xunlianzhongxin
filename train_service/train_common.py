# -*- coding: utf-8 -*-
"""
训练中心公共底座（train_common）——被 train_api / train_gpt / train_rvc 共用。
==============================================================
只放与具体引擎无关的东西：路径常量、运行模式、任务状态与队列容器、
标字校对/停止标志的存取器、子进程包装执行、素材标准化、时长估算、
基座模型下载、中间产物清理。

分层（依赖只朝一个方向，改代码互不影响）：
    train_common  ←  train_gpt / train_rvc  ←  train_api（页面与 API）
改某个引擎的训练逻辑去对应模块；改公共服务（队列/上传/页面）去 train_api。
本模块不 import FastAPI/uvicorn（引擎模块保持与 Web 框架无关）。
"""

import logging
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.request

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
# `-m 模块` 版包装：便携 Python 隔离模式忽略 PYTHONPATH，模块调用也必须用 -c 注入 sys.path
# （RVC 训练链全是 -m train.xxx 调用；run_module 在 __main__ 方式下自动把 sys.argv[0] 换成模块文件路径）
MODULE_WRAPPER = ("import os,sys,runpy; [sys.path.insert(0,p) for p in reversed(sys.argv[1].split(os.pathsep)) if p]; "
                  "sys.argv=sys.argv[2:]; runpy.run_module(sys.argv[0], run_name='__main__')")
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
_server = None  # uvicorn Server 对象由 train_api 启动后回填（C._server = server）

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
# 跨模块共享的"可变全局"一律通过下面的存取器访问，避免各模块 global 再绑定互相看不到
_review_evt = threading.Event()
_review_pending = {}  # {"name", "wavs":[完整路径], "items":[{"file","text"}], "inp_text"}

# 当前训练子进程与手动停止标志（供"停止训练"：杀进程树 + 标记停止）
_active_proc = None
_stop_requested = False


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    with _lock:
        _state["log"].append(line)
        _state["log"] = _state["log"][-800:]
    logger.info(msg)


def set_state(**kw):
    with _lock:
        _state.update(kw)


# ---- 停止标志 / 当前子进程存取器（train_api 的停止按钮、train_common 的进程执行共用） ----
def request_stop():
    global _stop_requested
    _stop_requested = True


def clear_stop():
    global _stop_requested
    _stop_requested = False


def is_stop_requested():
    return _stop_requested


def active_proc():
    return _active_proc


# ---- 标字校对存取器（train_gpt 的 _wait_review 等待、train_api 的校对接口读写） ----
def review_is_pending():
    return bool(_review_pending)


def review_start(name, wavs, items, inp_text):
    """进入标字校对等待状态（清事件 → 训练线程在 wait() 处暂停）。"""
    global _review_pending
    _review_pending = {"name": name, "wavs": wavs, "items": items, "inp_text": inp_text}
    _review_evt.clear()


def review_context():
    return _review_pending


def review_wait():
    """训练线程在校对等待点挂起，直到 review_release()（校对提交/跳过/停止）才继续。"""
    _review_evt.wait()


def review_release():
    """校对完成/跳过/停止：放行等待中的训练线程。"""
    _review_evt.set()


def review_clear():
    global _review_pending
    _review_pending = {}


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
    elif pythonpath and len(cmd) > 2 and str(cmd[0]) == PY312 and str(cmd[1]) == "-m":
        cmd = [PY312, "-c", MODULE_WRAPPER, pythonpath, str(cmd[2])] + [str(x) for x in cmd[3:]]
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
    """总目标步数（RVC 与页面估算共用）：默认 8000 步（10 分钟左右素材的社区推荐量，
    质量与耗时平衡；素材更短可 TRAIN_TARGET_STEPS 覆盖调小防过拟合）。"""
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


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


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
