# ============================================================
# 训练中心 ref 生成修复 - 自测
# 场景：构造一个 155 秒的超长"切片"，验证 _make_gpt_ref 强制裁剪到 ≤10 秒；
#       再验证 _ensure_gpt_ref_ok 对超长 ref 的兜底裁剪。
# 用法：py312 python tests\test_ref_fix.py
# ============================================================
import os
import sys
import shutil
import tempfile
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + os.sep + ".." + os.sep + "train_service")

import train_api  # noqa: E402
import librosa  # noqa: E402
import soundfile as sf  # noqa: E402

# 动态生成 155 秒超长 wav 充当"超长训练切片"（完全自包含，不依赖任何外部项目）
LONG_WAV = os.path.join(tempfile.gettempdir(), "ref_fix_long_155s.wav")
import numpy as np  # noqa: E402
sr = 32000
t = np.arange(int(155 * sr)) / sr
sf.write(LONG_WAV, (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr)
tmp = tempfile.mkdtemp(prefix="ref_fix_test_")
ok = True
try:
    work = os.path.join(tmp, "work")
    sliced = os.path.join(work, "sliced")
    ds = os.path.join(work, "fine_tune_dataset", "testrole")
    os.makedirs(sliced, exist_ok=True)
    os.makedirs(ds, exist_ok=True)
    # 1) 超长切片 + 标注文本
    shutil.copy(LONG_WAV, os.path.join(sliced, "seg_001.wav"))
    with open(os.path.join(ds, "2-name2text-1.txt"), "w", encoding="utf-8") as f:
        f.write("seg_001.wav\t\t\t这是一个超长切片的标注文本，用于测试参考音频裁剪逻辑是否正确。\n")
    # 2) 调用 _make_gpt_ref
    out_dir = os.path.join(tmp, "publish")
    os.makedirs(out_dir, exist_ok=True)
    train_api._make_gpt_ref(work, out_dir, "testrole")
    ref = os.path.join(out_dir, "ref.wav")
    rt = os.path.join(out_dir, "ref_text.txt")
    if not os.path.isfile(ref):
        print("[FAIL] _make_gpt_ref 未生成 ref.wav"); ok = False
    else:
        dur = librosa.get_duration(path=ref)
        txt = open(rt, encoding="utf-8").read().strip()
        print("_make_gpt_ref: ref.wav %.1f 秒, ref_text %d 字: %s" % (dur, len(txt), txt[:30]))
        if dur > 10:
            print("[FAIL] _make_gpt_ref 输出 ref.wav 仍超长 %.1f 秒" % dur); ok = False
        else:
            print("[OK] _make_gpt_ref 输出 ref.wav %.1f 秒（符合 3~10 秒）" % dur)
    # 3) _ensure_gpt_ref_ok 兜底：直接把超长 ref 放进去
    out2 = os.path.join(tmp, "publish2")
    os.makedirs(out2, exist_ok=True)
    shutil.copy(LONG_WAV, os.path.join(out2, "ref.wav"))
    with open(os.path.join(out2, "ref_text.txt"), "w", encoding="utf-8") as f:
        f.write("超长错乱文本" * 30)
    train_api._ensure_gpt_ref_ok(out2, "testrole2")
    dur2 = librosa.get_duration(path=os.path.join(out2, "ref.wav"))
    txt2 = open(os.path.join(out2, "ref_text.txt"), encoding="utf-8").read().strip()
    print("_ensure_gpt_ref_ok: 裁剪后 %.1f 秒, ref_text %d 字" % (dur2, len(txt2)))
    if dur2 > 10:
        print("[FAIL] _ensure_gpt_ref_ok 未裁剪超长 ref"); ok = False
    else:
        print("[OK] _ensure_gpt_ref_ok 兜底裁剪生效（%.1f 秒）" % dur2)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("RESULT:", "PASS" if ok else "FAIL")
