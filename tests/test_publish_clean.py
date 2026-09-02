# -*- coding: utf-8 -*-
r"""自测：解耦改造后 GPT 交付逻辑（交付模型文件夹）+ 发布后清理 work + RVC 清理逻辑。

不跑真实训练，用假权重/假切片验证：
1. _publish_gpt 生成 交付模型/角色（ckpt+pth+ref.wav+ref_text.txt 4件套）+ output 存档
2. _clean_work(name, "gpt") 清空该角色 work 中间产物
3. _clean_work(name, "rvc") 清空 rvc\logs\<角色>
跑法：runtime\py312\python.exe tests\test_publish_clean.py
"""
import os
import shutil
import struct
import sys
import wave
import math

os.environ["TRAIN_MODE"] = "wen_zi"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train_service"))
import train_api as ta  # noqa: E402

NAME = "zztest_xxx"


def make_wav(path, sec=3.0, sr=32000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = int(sec * sr)
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = b"".join(
            struct.pack("<h", int(12000 * math.sin(2 * math.pi * 440 * i / sr)))
            for i in range(n))
        w.writeframes(frames)


def main():
    print("== 运行模式:", ta.MODE_NAME, "引擎:", ta.MODE_ENGINE)
    assert ta.MODE_ENGINE == "gpt", "TRAIN_MODE=wen_zi 应固定 gpt 引擎"
    assert ta.DELIVERY_ROOT == os.path.join(ta.PROJECT_ROOT, "交付模型"), ta.DELIVERY_ROOT
    print("[OK] 常量：DELIVERY_ROOT =", ta.DELIVERY_ROOT)

    work = os.path.join(ta.WORK_ROOT, "gpt_" + NAME)

    # ---- 造模拟训练产物 ----
    sliced = os.path.join(work, "sliced")
    make_wav(os.path.join(sliced, "seg1.wav"))
    opt = os.path.join(work, "fine_tune_dataset", NAME)
    os.makedirs(opt, exist_ok=True)
    with open(os.path.join(opt, "2-name2text-0.txt"), "w", encoding="utf-8") as f:
        f.write("seg1.wav\tspk\tzh\t这是一段用于自测的参考文本\n")
    s1dir = os.path.join(ta.WORK_ROOT, "s1_half_weights", NAME)
    s2dir = os.path.join(ta.WORK_ROOT, "s2_weights", NAME)
    os.makedirs(s1dir, exist_ok=True)
    os.makedirs(s2dir, exist_ok=True)
    with open(os.path.join(s1dir, NAME + "-e1.ckpt"), "w") as f:
        f.write("fake-s1")
    with open(os.path.join(s2dir, NAME + "_e1_s8.pth"), "w") as f:
        f.write("fake-s2")

    # ---- 交付 ----
    ta._publish_gpt(NAME, work)
    out_dir = os.path.join(ta.DELIVERY_ROOT, NAME)
    pub_dir = os.path.join(ta.OUTPUT_ROOT, NAME)
    need = [os.path.join(out_dir, NAME + ".ckpt"), os.path.join(out_dir, NAME + ".pth"),
            os.path.join(out_dir, "ref.wav"), os.path.join(out_dir, "ref_text.txt"),
            os.path.join(pub_dir, NAME + ".ckpt"), os.path.join(pub_dir, NAME + ".pth")]
    for p in need:
        assert os.path.isfile(p), "交付产物缺失: %s" % p
    print("[OK] 交付：", out_dir, "（含 ckpt/pth/ref.wav/ref_text.txt 4件套）+ output 存档")

    # ---- 清理 work ----
    ta._clean_work(NAME, "gpt")
    for p in (work, s1dir, s2dir,
              os.path.join(ta.WORK_ROOT, "s1_logs", NAME),
              os.path.join(ta.WORK_ROOT, "s2_logs", NAME)):
        assert not os.path.exists(p), "清理失败: %s" % p
    assert os.path.isfile(os.path.join(out_dir, NAME + ".ckpt")), "交付模型被误删"
    print("[OK] work 中间产物已全部清理，交付模型保留")

    # ---- RVC 清理逻辑 ----
    exp = os.path.join(ta.RVC_ROOT, "logs", NAME)
    os.makedirs(exp, exist_ok=True)
    with open(os.path.join(exp, "G_999.pth"), "w") as f:
        f.write("fake")
    ta._clean_work(NAME, "rvc")
    assert not os.path.exists(exp), "RVC logs 清理失败"
    print("[OK] RVC 清理：rvc\\logs\\<角色> 已清除")

    # ---- 清理测试产物 ----
    shutil.rmtree(out_dir, ignore_errors=True)
    shutil.rmtree(pub_dir, ignore_errors=True)
    print("[OK] 测试产物已清理，全部自测通过")


if __name__ == "__main__":
    main()
