# -*- coding: utf-8 -*-
"""
训练中心（独立项目，端口 8050）— Web 服务与 API 层（完全自包含）
==============================================================
两种训练模式（TRAIN_MODE）：
  huan_sheng ：RVC 换音色训练 → 生成交付包（交付模型\\rvc\\<角色>\\，.pth + .index）
  wen_zi     ：GPT-SoVITS 文字驱动 TTS 训练 → 生成交付包（交付模型\\<角色>\\，ckpt+pth+ref.wav+ref_text.txt）

模块分层（依赖只朝一个方向，改代码互不影响）：
  train_api.py    —— 本文件：页面、API 路由、任务队列调度（改公共服务只动这里）
  train_common.py —— 公共底座：路径/状态/进程执行/素材标准化/清理/下载
  train_gpt.py    —— 文字驱动训练管线（ASR 标字/校对/S1/S2/交付）
  train_rvc.py    —— 换声训练管线（切分/特征/训练/索引/推理权重/交付）

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

import os
import re
import shutil
import subprocess
import sys
import threading
import traceback

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# 便携 Python 是 _pth 隔离模式（safe_path=True），不会自动把脚本目录加进 sys.path，
# 必须显式插入本目录才能 import 同目录的 train_common / train_gpt / train_rvc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train_common as C  # 仅供 _server 回填等少数场景用

# ---- 公共底座：路径 / 运行模式 / 状态与队列容器 / 存取器 / 公共函数 ----
from train_common import (
    AUTO_EXIT, DATA_ROOT, DELIVERY_ROOT, GPT_BASE, GSV_ROOT, MODE_DESC,
    MODE_ENGINE, MODE_NAME, OUTPUT_ROOT, PORT, PROJECT_ROOT, PY312,
    RVC_BASE, RVC_DELIVERY_DIR, RVC_ROOT, TRAIN_MODE, WORK_ROOT,
    _auto_batches, _cancel_exit, _clean_work, _gpu_mem_gb, _lock,
    _queue, _queue_done, _queue_lock, _state, _target_steps_by_data,
    active_proc, clear_stop, download_base, estimate_training,
    is_stop_requested, list_audio, log, logger, request_stop,
    review_context, review_is_pending, review_release, schedule_exit, set_state,
)

# ---- 引擎管线（从模块导入；旧名字在 train_api 上继续可用，tests 直接 import train_api 不受拆分影响） ----
from train_rvc import extract_infer_weights, train_rvc  # noqa: F401
from train_gpt import (  # noqa: F401
    train_gpt, _publish_gpt, _make_gpt_ref, _ensure_gpt_ref_ok,
    _s1_target_steps, _asr_text_ok, _asr_slices, _reslice_long,
    _wait_review, _merge_gpt_parts, _make_s1_config, _make_s2_config,
    _latest_recursive,
)

# ---------------- FastAPI ----------------
app = FastAPI(title="训练中心")

# 页面模板独立维护在 index_template.html（整体优化 + 任务队列 UI）
_TPL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index_template.html")
INDEX_HTML = open(_TPL, encoding="utf-8").read()


def _start_job(kind, name, epochs, clean=True, raw_dir=None, review=False):
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
        clear_stop()
        _state.update(running=True, ok=False, error="", log=[], engine=kind, name=name, step="启动")
    def worker():
        ok, err = _execute_task(kind, name, epochs, clean, raw_dir, review=review)
        if ok:
            set_state(ok=True, step="完成", running=False)
        else:
            if is_stop_requested():
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
        if is_stop_requested():
            return False, "已手动停止"
        traceback.print_exc()
        return False, str(exc)


def _after_task_done():
    """当前任务结束后调用：有队列任务则继续跑下一个，否则所有任务完成后自动退出。
    注意：schedule_exit 不能在本函数持锁时调用（内部也要拿 _queue_lock，会死锁）。"""
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
    clear_stop()
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
                if is_stop_requested():
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
    queue=True 时只保存不启动训练（前端随后加入任务队列），返回保存目录。
    engine 参数仅为兼容前端保留，实际强制用当前模式的引擎（解耦：换声模式只训 RVC、文字驱动只训 GPT）。"""
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
def download_base_api(req: dict):
    """下载当前模式的基座模型（已存在的文件自动跳过）。请求体可为 {}。"""
    return _start_job("download_" + MODE_ENGINE, "", 0)


@app.get("/api/review_status")
def review_status_api():
    """查询当前是否有待校对的标字任务（含每段识别文字）。"""
    if not review_is_pending():
        return {"pending": False}
    ctx = review_context()
    return {"pending": True, "name": ctx["name"],
            "items": [dict(it) for it in ctx["items"]]}


@app.post("/api/review_submit")
def review_submit_api(req: dict):
    """提交校对后的文字（items=[{file, text}, ...]），用校对结果重写标注并继续训练。"""
    if not review_is_pending():
        return JSONResponse({"detail": "当前没有待校对任务"}, status_code=400)
    items = req.get("items") or []
    if not items:
        return JSONResponse({"detail": "没有提交内容"}, status_code=400)
    ctx = review_context()
    text_by_file = {str(it.get("file", "")): str(it.get("text") or "").strip() for it in items}
    with open(ctx["inp_text"], "w", encoding="utf-8") as f:
        for w in ctx["wavs"]:
            text = text_by_file.get(os.path.basename(w), "")
            if text:
                f.write("%s|0|zh|%s\n" % (w, text))
    log("标字校对完成：%d 段已确认，继续训练" % len(ctx["wavs"]))
    review_release()
    return {"message": "校对完成，继续训练"}


@app.post("/api/review_skip")
def review_skip_api():
    """跳过校对，直接用原始识别文字继续训练。"""
    if not review_is_pending():
        return JSONResponse({"detail": "当前没有待校对任务"}, status_code=400)
    log("跳过标字校对，使用原始识别文字继续训练")
    review_release()
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
    with _lock:
        if not _state["running"]:
            return JSONResponse({"detail": "当前没有正在训练的任务"}, status_code=400)
        request_stop()
    # 若处于标字校对暂停阶段，放行事件让任务走到下一个训练步骤时被停止标志拦下
    if review_is_pending():
        review_release()
    p = active_proc()
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
    C._server = _server  # 自动退出（schedule_exit）通过它请求 uvicorn 停止
    _server.run()
