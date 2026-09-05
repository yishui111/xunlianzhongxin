# -*- coding: utf-8 -*-
"""
GPT-SoVITS 文字驱动训练管线（TRAIN_MODE=wen_zi）。
==============================================================
素材标准化 → 切片（超长切片静音处二次切分）→ ASR 标字（幻觉标注过滤）→
标字校对（可暂停等页面核对）→ 文本/HuBERT/语义特征 → S1/S2 训练 →
发布交付包（交付模型\\<角色>\\，ckpt+pth+ref.wav+ref_text.txt 四件套）→ 清理中间产物。
只依赖 train_common（公共底座），不依赖 Web 层（train_api）；
改文字驱动训练逻辑只动这个文件，换声（train_rvc）完全不受影响。
"""

import json
import os
import re
import shutil

from train_common import (
    DATA_ROOT, DELIVERY_ROOT, GPT_BASE, GSV_PYTHONPATH, GSV_ROOT, OUTPUT_ROOT,
    PROJECT_ROOT, PY312, WORK_ROOT,
    list_audio, glob_wavs, log, review_clear, review_start, review_wait,
    run_proc, run_proc_env, set_state,
    _audio_dur, _auto_batches, _cuda_available, _clean_work, _prep_audio,
)


def _s1_target_steps(total_sec):
    """GPT S1（文本→语义 AR）目标步数：随素材时长缩放，上限压在过拟合阈值以下。

    漏字根因（keai/laolei 实测定位）：旧默认 8000 步在百来个切片的小数据集上折算
    400~600 轮，S1 把训练集背下来后对齐变脆，合成吞字；而文字驱动项目自带的
    Ayaka/Azhong（从不漏字）是官方默认 15 轮 ≈ 几百步。音色像不像主要由 S2+参考
    音频决定，S1 只负责韵律/咬字，按官方量级训练即可：
    每分钟素材约 100 步，下限 500（再少韵律不稳），上限 1200（防过拟合漏字）。
    TRAIN_GPT_TARGET_STEPS（模式专属）或旧变量 TRAIN_TARGET_STEPS 可强制覆盖。"""
    env = os.environ.get("TRAIN_GPT_TARGET_STEPS") or os.environ.get("TRAIN_TARGET_STEPS")
    if env:
        return int(env)
    minutes = max(total_sec / 60.0, 1.0)
    return int(min(1200, max(500, minutes * 100)))


def _reslice_long(sliced, max_sec=12, target_sec=8):
    """把超长切片（>max_sec 秒）二次切成 target_sec 秒左右的段。
    切片脚本按静音切，素材连续说话无停顿时会切出超长段（60~96 秒），
    导致训练特征巨大、计算爆炸、训练卡死。
    切点在目标位置附近找最安静处落刀，避免硬切把词从中间切断——
    残缺词配错误标注会教坏 S1 对齐，合成漏字。"""
    import numpy as np
    import soundfile as sf

    def _quiet_cut(y, lo, hi, sr):
        """在 [lo, hi)（采样点）里找 100ms 窗口能量最低处，返回该窗口中点。"""
        frame = max(1, int(sr * 0.1))
        lo, hi = max(0, int(lo)), min(len(y), int(hi))
        if hi - lo <= frame:
            return (lo + hi) // 2
        best_i, best_e = lo, None
        for i in range(lo, hi - frame + 1, max(1, frame // 2)):
            e = float(np.mean(np.asarray(y[i:i + frame], dtype=np.float64) ** 2))
            if best_e is None or e < best_e:
                best_e, best_i = e, i
        return best_i + frame // 2

    cut = 0
    removed = 0
    for w in list_audio(sliced):
        try:
            info = sf.info(w)
            dur = info.frames / info.samplerate
            if dur <= max_sec:
                continue
            y, sr = sf.read(w, dtype="float32")
            if y.ndim > 1:
                y = y[:, 0]
            base = os.path.splitext(os.path.basename(w))[0]
            min_seg = int(2.0 * sr)
            bounds = [0]
            pos = 0
            while len(y) - pos > int(max_sec * sr):
                ideal = pos + int(target_sec * sr)
                c = _quiet_cut(y, ideal - int(1.5 * sr), ideal + int(1.5 * sr), sr)
                if c - bounds[-1] < min_seg:
                    c = pos + int(target_sec * sr)
                bounds.append(min(c, len(y)))
                pos = bounds[-1]
            bounds.append(len(y))
            idx = 0
            for s, e in zip(bounds, bounds[1:]):
                if e - s < min_seg:  # 尾部不足 2 秒丢弃（太短无训练价值）
                    continue
                sf.write(os.path.join(sliced, "%s_s%d.wav" % (base, idx)), y[s:e], sr)
                idx += 1
            if idx > 0:
                os.remove(w)
                removed += 1
                cut += idx
                log("超长切片 %s（%.0f 秒）→ 按最安静处切成 %d 段" % (os.path.basename(w), dur, idx))
        except Exception:
            pass
    if removed:
        log("已把 %d 个超长切片二次切分为 %d 段（切点取最安静处，不截断词）" % (removed, cut))
    return cut


def _asr_text_ok(text, dur):
    """训练标注体检。SenseVoice 对噪声/杂音会幻觉出重复文字、给静音编字，这类错字
    标签会教坏 S1 的文本→语义对齐，是合成漏字/错字的直接原因之一，不进训练集。
    正常中文语速 ≤6 字/秒，>7 字/秒基本是幻觉标签。返回 (是否可用, 原因)。"""
    n = len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", text))
    if n < 2:
        return False, "有效字数<2"
    if dur > 0 and n / dur > 7.0:
        return False, "字/秒=%.1f 疑似ASR幻觉" % (n / dur)
    return True, ""


def _asr_slices(wavs, out_path):
    """SenseVoice 把切片语音转成文字，写入 GPT-SoVITS 需要的 input_list（path|spk|lang|text）。
    幻觉/碎片段标注自动丢弃（_asr_text_ok），防错字标签教坏模型。"""
    try:
        from funasr import AutoModel
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("ASR 依赖 funasr 不可用：%s" % exc)
    asr_dir = os.path.join(PROJECT_ROOT, "gptsovits", "asr", "SenseVoiceSmall")
    device = "cuda" if _cuda_available() else "cpu"
    model = AutoModel(model=asr_dir, trust_remote_code=False, device=device, disable_update=True)
    lines = []
    n_drop = 0
    for w in wavs:
        try:
            res = model.generate(input=w)[0]
            text = re.sub(r"<\|[^|]*\|>", "", res.get("text", "")).strip()
        except Exception as exc:  # noqa: BLE001
            log("ASR 失败 %s: %s" % (os.path.basename(w), str(exc)[:120]))
            continue
        ok, why = _asr_text_ok(text, _audio_dur(w))
        if not ok:
            n_drop += 1
            log("丢弃可疑切片 %s（%s）：%s" % (os.path.basename(w), why, text[:30]))
            continue
        lines.append("%s|0|zh|%s" % (w, text))
    if n_drop:
        log("已丢弃 %d 个可疑标注切片（幻觉/碎片段不进训练集，防合成漏字）" % n_drop)
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
    review_start(name, wavs, items, inp_text)
    set_state(step="标字校对中（%d 段，请核对识别文字；普通话可跳过，方言务必逐句校对）" % len(items))
    log("▶ 标字校对：请在页面核对/修改识别文字后点“确认继续”；普通话识别准确可点“跳过校对”")
    review_wait()
    review_clear()


def _make_s1_config(path, opt_dir, name, epochs, batch=8):
    """生成完整的 S1 训练配置（仓库自带 yaml 缺训练必需字段，这里从完整模板生成）。
    batch 由 _auto_batches 按显卡显存自动选择。"""
    import yaml
    cfg = {
        "train": {
            "seed": 1234,
            "epochs": int(epochs),
            "batch_size": int(batch),
            # 保存频率：控制在全程约 15 次左右（保存时 Lightning 先在内存序列化 0.9GB+，
            # 次数太多容易触发 MemoryError；本机 32GB 内存实测每 14 epoch 保存会失败）。
            # S1 步数改小后总轮数只有几十轮，按 epochs//15 保证少存多保
            "save_every_n_epoch": max(1, int(epochs) // 15),
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

    # 训练量自动规划：S1 随素材时长缩放（约 100 步/分钟，上限 1200——旧默认 8000 步会让 S1
    # 在小数据集上过拟合，合成漏字；TRAIN_GPT_TARGET_STEPS 可覆盖）；S2 默认 4000 步保持充足
    # （音色像不像主要靠 S2，可 TRAIN_S2_TARGET_STEPS 覆盖）——S2 每轮有两次大权重落盘，步数太多耗时很长
    total_sec = sum(_audio_dur(a) for a in audios)
    target_steps = _s1_target_steps(total_sec)
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
