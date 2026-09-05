# -*- coding: utf-8 -*-
"""
RVC 换声训练管线（TRAIN_MODE=huan_sheng）。
==============================================================
数据切分 → F0/HuBERT 特征 → 训练 → 索引 → 提取推理权重 →
交付（交付模型\\rvc\\<角色>\\，.pth + .index）→ 清理中间产物。
只依赖 train_common（公共底座），不依赖 Web 层（train_api）；
改换声训练逻辑只动这个文件，文字驱动（train_gpt）完全不受影响。
"""

import os
import re
import shutil

from train_common import (
    PY312, DATA_ROOT, OUTPUT_ROOT, RVC_BASE, RVC_DELIVERY_DIR, RVC_ROOT,
    list_audio, glob_wavs, latest_glob, log, run_proc, set_state,
    _audio_dur, _auto_batches, _clean_work, _prep_audio, _target_steps_by_data,
)


def _rvc_target_steps(total_sec):
    """RVC 总步数目标：TRAIN_RVC_TARGET_STEPS 优先（模式专属变量），其次旧变量
    TRAIN_TARGET_STEPS（兼容既有部署），默认 8000 步。
    RVC 训多不会漏字（漏字是 GPT S1 过拟合的问题），8000 步是社区常用量。"""
    env = os.environ.get("TRAIN_RVC_TARGET_STEPS") or os.environ.get("TRAIN_TARGET_STEPS")
    return int(env) if env else 8000


def extract_infer_weights(g_file, name, exp_dir):
    """从训练检查点 logs/<角色>/G_*.pth 提取推理格式权重到 assets/weights/<角色>.pth。

    换声项目 rvc_service 的 vc.get_vc 只认推理格式（含 weight/config/f0/version 键，
    即 RVC 官方"提取模型"产物），直接拿训练检查点 G_*.pth 当角色权重会 KeyError。
    这里调用 RVC 官方 train.process_ckpt.savee 完成提取（fp16，去掉 enc_q）。
    """
    weights_dir = os.path.join(RVC_ROOT, "assets", "weights")
    os.makedirs(weights_dir, exist_ok=True)
    m = re.search(r"G_(\d+)\.pth$", g_file)
    step = m.group(1) if m else "0"
    # 路径一律走 argv（sys.argv），不做代码字符串内插，避免特殊字符破坏 -c 代码
    code = "\n".join([
        "import sys, torch",
        "_, rvc_root, g_file, cfg, name, step = sys.argv[:6]",
        "sys.path.insert(0, rvc_root)",
        "from train.process_ckpt import savee",
        "from train.utils import get_hparams_from_file",
        "cpt = torch.load(g_file, map_location='cpu')",
        "hps = get_hparams_from_file(cfg)",
        "print('savee:', savee(cpt['model'], '48k', 1, name, step, 'v2', hps))",
    ])
    run_proc([PY312, "-c", code, RVC_ROOT, g_file,
              os.path.join(exp_dir, "config.json"), name, step],
             RVC_ROOT, "提取推理权重", pythonpath=RVC_ROOT)
    out = os.path.join(weights_dir, name + ".pth")
    if not os.path.isfile(out):
        raise RuntimeError("推理权重提取失败：%s 未生成（savee 输出见上方日志）" % out)
    log("推理权重已提取：%s（fp16，可直接被换声项目 rvc_service 加载）" % out)
    return out


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
    # noparallel 传 True（串行）：数据集都是单/少文件多进程无收益，且 Windows spawn
    # 无法 pickle runpy 包装环境里的 __main__.PreProcess 类（并行必挂）
    run_proc([PY312, "-m", "train.preprocess",
              raw, "48000", "8", exp, "True", "3.7"], RVC_ROOT, "RVC 数据切分", pythonpath=RVC_ROOT)

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
    target_steps = _rvc_target_steps(total_sec)
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
    # output 存档原始训练检查点（可用于续训；注意它是训练格式，不能直接给换声项目用）
    shutil.copy(g_file, os.path.join(out_dir, name + "_train_ckpt.pth"))
    run_proc([PY312, "-m", "train.train_index",
              name, "v2", "assets/indices", "8"], RVC_ROOT, "RVC 索引训练", pythonpath=RVC_ROOT)
    set_state(step="6/6 提取推理权重并生成交付包")
    # 交付包必须是推理格式权重（换声项目才能加载），不能直接用训练检查点
    infer_pth = extract_infer_weights(g_file, name, exp)
    deliv = os.path.join(RVC_DELIVERY_DIR, name)
    os.makedirs(deliv, exist_ok=True)
    shutil.copy(infer_pth, os.path.join(deliv, name + ".pth"))
    # 新版 train_index 的索引文件名带 added_IVF 前缀（<名>_added_IVF.._<名>_v2.index），
    # 精确找 <名>.index 会静默漏掉；按角色名模糊匹配取最新
    import glob as _glob
    idx_hits = sorted(
        _glob.glob(os.path.join(RVC_ROOT, "assets", "indices", "*%s*.index" % name)),
        key=os.path.getmtime)
    if idx_hits:
        shutil.copy(idx_hits[-1], os.path.join(deliv, name + ".index"))
    else:
        log("警告：未找到 %s 的索引文件（.index），交付包将不含特征检索索引（相似度略降，不影响可用）" % name)
    log("模型已交付：%s → 交付模型\\rvc\\%s（%s）" % (name, name, deliv))
    log("使用方法：把该交付文件夹整个复制到 换声项目 rvc_service\\models\\ 下，刷新工作台即自动出现（只换音色，不改音高语调）")
    _clean_work(name, "rvc")
