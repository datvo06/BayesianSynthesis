from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from PIL import Image

from kernel_viz import render_kernel_chain, render_kernel_graph, render_kernel_mutation


@dataclass
class KernelNode:
    name: str | None = None
    op: str | None = None
    left: "KernelNode | None" = None
    right: "KernelNode | None" = None

    def pretty(self) -> str:
        if self.op and self.left and self.right:
            return f"({self.left.pretty()} {self.op} {self.right.pretty()})"
        return self.name or "?"


def _strip_highlight(s: str) -> str:
    return s.replace("[[", "").replace("]]", "").strip()


def _skip_ws(s: str, i: int) -> int:
    while i < len(s) and s[i].isspace():
        i += 1
    return i


def _parse_base(s: str, i: int) -> tuple[KernelNode, int]:
    start = i
    while i < len(s) and s[i] not in [" ", ")", "+", "*"]:
        if s[i] == "(":
            break
        i += 1
    if i < len(s) and s[i] == "(":
        depth = 0
        while i < len(s):
            if s[i] == "(":
                depth += 1
            elif s[i] == ")":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            i += 1
    token = s[start:i].strip()
    if not token:
        return KernelNode(name="X"), min(i + 1, len(s))
    return KernelNode(name=token), i


def _parse_expr(s: str, i: int = 0) -> tuple[KernelNode, int]:
    i = _skip_ws(s, i)
    if i >= len(s):
        return KernelNode(name="?"), i
    if s[i] == "(":
        i += 1
        left, i = _parse_expr(s, i)
        i = _skip_ws(s, i)
        op = s[i]
        i += 1
        right, i = _parse_expr(s, i)
        i = _skip_ws(s, i)
        if i < len(s) and s[i] == ")":
            i += 1
        return KernelNode(op=op, left=left, right=right), i
    return _parse_base(s, i)


def parse_kernel_expr(s: str) -> KernelNode:
    clean = _strip_highlight(s)
    node, _ = _parse_expr(clean, 0)
    return node


def parse_log(path: str) -> List[Dict]:
    rounds: List[Dict] = []
    current: Dict | None = None
    step: Dict | None = None

    step_re = re.compile(r"Step\s+(\d+)\s+\|\s+(accept|reject)\s+\|\s+log_alpha=([-\d\.]+)")

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("Init: "):
                if current:
                    rounds.append(current)
                current = {"init": line[len("Init: ") :], "steps": []}
                step = None
                continue
            m = step_re.match(line)
            if m:
                step = {
                    "idx": int(m.group(1)),
                    "status": m.group(2),
                    "log_alpha": float(m.group(3)),
                }
                current["steps"].append(step)
                continue
            if line.startswith("Sever path: "):
                step["sever_path"] = line[len("Sever path: ") :].strip()
                continue
            if line.startswith("Before: "):
                step["before"] = line[len("Before: ") :].strip()
                continue
            if line.startswith("After : "):
                step["after"] = line[len("After : ") :].strip()
                continue

    if current:
        rounds.append(current)
    return rounds


def _path_to_tuple(s: str) -> Tuple[str, ...]:
    if s == "root" or s == "":
        return ()
    return tuple(s)


def visualize_log_run(log_path: str, out_dir: str) -> None:
    rounds = parse_log(log_path)
    os.makedirs(out_dir, exist_ok=True)

    for r_idx, r in enumerate(rounds, start=1):
        steps_sorted = sorted(r["steps"], key=lambda item: item["idx"])
        all_scores = [step["log_alpha"] for step in steps_sorted]
        y_min, y_max = min(all_scores), max(all_scores)
        y_pad = (y_max - y_min) * 0.1 if y_max > y_min else 1.0
        y_limits = (y_min - y_pad, y_max + y_pad)

        init_kernel = parse_kernel_expr(r["init"])
        chain_kernels = [init_kernel]
        chain_scores = [None]
        render_kernel_graph(
            init_kernel,
            title=f"Round {r_idx} init",
            save_path=os.path.join(out_dir, f"round_{r_idx:02d}_init.png"),
        )

        frame_scores: list[float] = []
        frames: list[Image.Image] = []
        mutation_paths: list[str] = []

        for st in steps_sorted:
            idx = st["idx"]
            status = st["status"]
            log_alpha = st["log_alpha"]
            sever_path = _path_to_tuple(st.get("sever_path", ""))
            before_k = parse_kernel_expr(st["before"])
            after_k = parse_kernel_expr(st["after"])

            prefix = f"round_{r_idx:02d}_step_{idx:03d}_{status}"
            render_kernel_graph(
                before_k,
                highlight_paths={sever_path: "#ff6666"},
                base_fillcolor="#ffffff",
                score=log_alpha,
                title=f"Round {r_idx} step {idx:03d} before",
                save_path=os.path.join(out_dir, f"{prefix}_before.png"),
            )
            render_kernel_graph(
                after_k,
                highlight_paths={sever_path: "#66cc66"},
                base_fillcolor="#ffffff",
                score=log_alpha,
                title=f"Round {r_idx} step {idx:03d} after",
                save_path=os.path.join(out_dir, f"{prefix}_after.png"),
            )

            render_kernel_mutation(
                before_k,
                after_k,
                sever_path=sever_path,
                score=log_alpha,
                title=f"Round {r_idx} step {idx:03d} mutation",
                save_path=os.path.join(out_dir, f"{prefix}_mutation.png"),
            )
            mutation_paths.append(os.path.join(out_dir, f"{prefix}_mutation.png"))
            frame_scores.append(log_alpha)

            if len(chain_kernels) < 5:
                chain_kernels.append(after_k if status == "accept" else before_k)
                chain_scores.append(log_alpha)

        if len(chain_kernels) >= 2:
            render_kernel_chain(
                chain_kernels,
                chain_scores,
                title=f"Round {r_idx} chain (first {len(chain_kernels)} steps)",
                save_path=os.path.join(out_dir, f"round_{r_idx:02d}_chain_5.png"),
            )

        if mutation_paths:
            mutation_imgs = [Image.open(p).convert("RGBA") for p in mutation_paths]
            max_mut_w = max(img.width for img in mutation_imgs)
            max_mut_h = max(img.height for img in mutation_imgs)

            for i, mut_img in enumerate(mutation_imgs):
                mut_img = _pad_to_size(mut_img, max_mut_w, max_mut_h)
                score_img = _render_score_plot(
                    frame_scores[: i + 1],
                    title="Score by step",
                    total_steps=len(steps_sorted),
                    y_limits=y_limits,
                )
                frame = _stack_images_vertically(mut_img, score_img)
                frames.append(frame)

        if frames:
            max_w = max(img.width for img in frames)
            max_h = max(img.height for img in frames)
            frames = [_pad_to_size(img, max_w, max_h) for img in frames]
            gif_path = os.path.join(out_dir, f"round_{r_idx:02d}_mutation.gif")
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=900,
                loop=0,
            )


def _render_score_plot(
    scores: list[float],
    *,
    title: str,
    total_steps: int,
    y_limits: tuple[float, float],
) -> Image.Image:
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot(range(len(scores)), scores, color="#333333")
    ax.scatter([len(scores) - 1], [scores[-1]], color="#cc0000", s=30)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Step")
    ax.set_ylabel("Score")
    ax.set_xlim(-0.5, max(1, total_steps - 0.5))
    ax.set_ylim(*y_limits)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")


def _stack_images_vertically(top: Image.Image, bottom_img: Image.Image) -> Image.Image:
    if bottom_img.width != top.width:
        new_height = int(bottom_img.height * (top.width / bottom_img.width))
        bottom_img = bottom_img.resize((top.width, new_height))
    stacked = Image.new("RGBA", (top.width, top.height + bottom_img.height), (255, 255, 255, 255))
    stacked.paste(top, (0, 0))
    stacked.paste(bottom_img, (0, top.height))
    return stacked


def _pad_to_size(img: Image.Image, width: int, height: int) -> Image.Image:
    if img.width == width and img.height == height:
        return img
    padded = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    x = (width - img.width) // 2
    y = (height - img.height) // 2
    padded.paste(img, (x, y))
    return padded



if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(here, "assets", "log_run.txt")
    out_dir = os.path.join(here, "assets", "log_run_viz")
    visualize_log_run(log_path, out_dir)
