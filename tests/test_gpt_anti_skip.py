# -*- coding: utf-8 -*-
r"""自测：防合成漏字三件套（不跑真实训练）。

1. S1 步数随素材时长自适应且有上限（旧默认 8000 步在小数据集上过拟合 → 合成漏字）
2. ASR 标注体检：幻觉/碎片段标注被拒收，不进训练集
3. 超长切片二次切分时在最安静处下刀，不把词从中间切断

跑法：runtime\py312\python.exe tests\test_gpt_anti_skip.py
"""
import os
import shutil
import sys
import tempfile

os.environ["TRAIN_MODE"] = "wen_zi"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train_service"))
import train_api as ta  # noqa: E402


def main():
    import numpy as np
    import soundfile as sf

    print("== 1) S1 步数随素材时长缩放且有上限（防过拟合漏字）")
    os.environ.pop("TRAIN_TARGET_STEPS", None)
    assert ta._s1_target_steps(180) == 500, ta._s1_target_steps(180)      # 3 分钟 → 下限
    assert ta._s1_target_steps(600) == 1000, ta._s1_target_steps(600)     # 10 分钟 → 100 步/分钟
    assert ta._s1_target_steps(1800) == 1200, ta._s1_target_steps(1800)   # 30 分钟 → 上限
    assert ta._target_steps_by_data(600) == 8000                          # RVC 共用函数不受影响
    os.environ["TRAIN_GPT_TARGET_STEPS"] = "900"
    assert ta._s1_target_steps(600) == 900                                # 模式前缀变量优先
    os.environ.pop("TRAIN_GPT_TARGET_STEPS")
    os.environ["TRAIN_TARGET_STEPS"] = "8000"
    assert ta._s1_target_steps(600) == 8000                               # 显式覆盖仍生效
    os.environ.pop("TRAIN_TARGET_STEPS", None)

    print("== 2) ASR 标注体检：幻觉/碎片段被拒")
    ok, why = ta._asr_text_ok("今天天气不错我们去公园里走走吧", 5.0)
    assert ok, why
    ok, _ = ta._asr_text_ok("好", 3.0)
    assert not ok, "单字切片应被拒"
    ok, why = ta._asr_text_ok("你好吗" * 20, 5.0)
    assert not ok and "幻觉" in why, why

    print("== 3) 超长切片按最安静处二次切分（不截断词）")
    sr = 32000
    tmp = tempfile.mkdtemp(prefix="gpt_anti_skip_")
    try:
        # 7 秒大声 + 0.5 秒安静，重复 5 次 = 37.5 秒；安静区起点在 7.0 + 7.5k 秒
        # （soundfile 写 16-bit WAV 时 float 数据须在 ±1 内，否则削波响/静不分）
        t = np.arange(int(7 * sr)) / sr
        loud = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        t2 = np.arange(int(0.5 * sr)) / sr
        quiet = (0.005 * np.sin(2 * np.pi * 440 * t2)).astype(np.float32)
        y = np.concatenate([loud, quiet] * 5)
        src = os.path.join(tmp, "long.wav")
        sf.write(src, y, sr)

        ta._reslice_long(tmp)

        assert not os.path.isfile(src), "原超长切片应被替换"
        pieces = sorted(f for f in os.listdir(tmp) if f.startswith("long_s"))
        assert 4 <= len(pieces) <= 6, "应切成约 5 段，实际 %d" % len(pieces)
        bounds = []
        total = 0.0
        for f in pieces:
            info = sf.info(os.path.join(tmp, f))
            dur = info.frames / info.samplerate
            assert 2.0 <= dur <= 9.6, "%s 时长 %.2f 秒越界" % (f, dur)
            bounds.append(total)
            total += dur
        # 每个切点应落在安静区（7.0 + 7.5k 秒，容差 0.2 秒），而不是硬切的 8.0 秒处
        for b in bounds[1:]:
            phase = b % 7.5
            assert 6.8 <= phase <= 7.7, "切点 %.2f 未落在安静区（phase=%.2f）" % (b, phase)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("OK：防漏字三件套全部通过")


if __name__ == "__main__":
    main()
