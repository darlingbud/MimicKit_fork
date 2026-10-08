#!/usr/bin/env python3
"""
把 rl-infra 格式的 motion.npz 转成 MimicKit 的 .pkl 格式。

用法:

    # 传目录（目录内有 motion.npz）
    python tools/npz_to_pkl_go2.py /path/to/assets/motions/<motion_id>

    # 或直接传 npz 文件
    python tools/npz_to_pkl_go2.py /path/to/motion.npz

    # 指定输出路径 / loop 模式
    python tools/npz_to_pkl_go2.py <src> -o data/motions/go2/xxx.pkl --loop-mode 1

默认输出: <repo>/data/motions/go2/<motion_id>.pkl

(motion_id 取源目录名，若直接传 npz 文件则取其父目录名。)

注意：rl-infra 里文件夹名可能叫 "panda_*"，但 joint_names 是
     FR/FL/RR/RL_hip/thigh/calf_joint，实际是 Unitree GO2 四足机器人 (12 DOF)，
     所以输出统一加 "go2_" 前缀。
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

# <repo>/tools/npz_to_pkl_go2.py -> <repo>
REPO_ROOT = Path(__file__).resolve().parents[1]

# MimicKit LoopMode: CLAMP = 0, WRAP = 1
LOOP_MODES = {"clamp": 0, "wrap": 1}


def quat_wxyz_to_expmap(quat_wxyz: np.ndarray) -> np.ndarray:
    """wxyz 四元数 → 3D 指数映射 (axis-angle)。

    expmap = axis * angle
    其中 q = (w, x, y, z)，angle = 2*acos(w)
    """
    w, x, y, z = quat_wxyz
    # 数值稳定：clamp w 到 [-1, 1]
    w = np.clip(w, -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    sin_half = np.sin(angle / 2.0)

    # 避免除零
    if sin_half < 1e-6:
        # 小角度：expmap ≈ 2 * (x, y, z)
        return 2.0 * np.array([x, y, z], dtype=np.float32)

    axis = np.array([x, y, z]) / sin_half
    return (axis * angle).astype(np.float32)


def resolve_src(src: str) -> Path:
    """把传入的路径解析成 motion.npz 文件路径。"""
    p = Path(src).expanduser()
    if p.is_dir():
        p = p / "motion.npz"
    if not p.is_file():
        raise FileNotFoundError(f"找不到 npz 文件: {p}")
    return p.resolve()


def parse_loop_mode(value: str) -> int:
    v = str(value).strip().lower()
    if v in LOOP_MODES:
        return LOOP_MODES[v]
    try:
        iv = int(v)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--loop-mode 需要 0/1 或 clamp/wrap，收到: {value!r}"
        )
    if iv not in (0, 1):
        raise argparse.ArgumentTypeError(f"--loop-mode 只能是 0 或 1，收到: {value!r}")
    return iv


def convert(src_npz: Path, out_pkl: Path, loop_mode: int) -> None:
    out_pkl.parent.mkdir(parents=True, exist_ok=True)

    d = np.load(str(src_npz), allow_pickle=True)

    fps = float(d["fps"])
    timestamps = d["timestamps"]
    joint_names = list(d["joint_names"])
    joint_pos = d["joint_pos"]            # (T, 12)
    body_pos_w = d["body_pos_w"]          # (T, N, 3)
    body_quat_w = d["body_quat_w"]        # (T, N, 4) -- base is body[0]
    body_names = list(d["body_names"])
    n_frames = timestamps.shape[0]

    print(f"Loaded npz: {src_npz}")
    print(f"  frames: {n_frames} @ {fps} fps ({n_frames / fps:.2f}s)")
    print(f"  joint_names: {joint_names}")
    print(f"  body_names: {body_names}")

    # 断言 base body 在 body_pos_w/body_quat_w 第一位
    assert body_names[0] == "base", "base body 必须在 body_names 首位"

    n_joints = joint_pos.shape[1]
    frames = []
    for t in range(n_frames):
        # root_pos: base 的世界坐标位置
        root_pos = body_pos_w[t, 0].astype(np.float32)

        # root_rot: base 的世界坐标四元数 → expmap
        root_quat = body_quat_w[t, 0]
        root_rot = quat_wxyz_to_expmap(root_quat)

        # joint_dof: 12 个关节角度
        joint_dof = joint_pos[t].astype(np.float32)

        # 拼成 (3 + 3 + 12) 维: [root_pos, root_rot, joint_dof]
        frame = np.concatenate([root_pos, root_rot, joint_dof]).tolist()
        frames.append(frame)

    # MimicKit format: dict with loop_mode, fps, frames
    out_dict = {
        "loop_mode": loop_mode,
        "fps": fps,
        "frames": frames,
    }

    with open(out_pkl, "wb") as f:
        pickle.dump(out_dict, f)

    mode_name = {0: "CLAMP", 1: "WRAP"}[loop_mode]
    print(f"Saved {out_pkl}")
    print(f"  frames: {len(frames)}")
    print(f"  per-frame dim: {len(frames[0])}")
    print(f"  loop_mode: {mode_name} ({loop_mode})")
    print(f"  fps: {fps}")
    print(f"  duration: {len(frames) / fps:.2f}s")

    # 一致性 sanity check
    assert len(frames) == joint_pos.shape[0]
    assert len(frames[0]) == 3 + 3 + n_joints


def main() -> int:
    parser = argparse.ArgumentParser(
        description="rl-infra motion.npz -> MimicKit .pkl",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "src",
        help="rl-infra motion 目录（含 motion.npz）或直接给 motion.npz 路径",
    )
    parser.add_argument(
        "-o", "--out",
        default=None,
        help="输出 pkl 路径；默认 <repo>/data/motions/go2/<motion_id>.pkl",
    )
    parser.add_argument(
        "--loop-mode",
        type=parse_loop_mode,
        default=1,
        help="循环模式: 1/wrap = 循环, 0/clamp = 播完停在末帧",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="输出文件名主干（不含 .pkl），默认由源目录名推导并加 go2_ 前缀",
    )
    args = parser.parse_args()

    src_npz = resolve_src(args.src)
    motion_id = src_npz.parent.name or src_npz.stem

    # rl-infra 里 GO2 的轨迹目录常被命名为 "panda_*"（历史原因），
    # 输出时需要去掉这个误导性的前缀。
    base_name = motion_id
    if base_name.startswith("panda_"):
        base_name = base_name[len("panda_"):]

    if args.name:
        stem = args.name
    elif base_name.startswith("go2"):
        stem = base_name
    else:
        stem = f"go2_{base_name}"

    out_pkl = (
        Path(args.out).expanduser().resolve()
        if args.out
        else (REPO_ROOT / "data" / "motions" / "go2" / f"{stem}.pkl")
    )

    convert(src_npz, out_pkl, args.loop_mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
