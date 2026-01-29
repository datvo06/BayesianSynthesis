from __future__ import annotations

from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import networkx as nx


def _get_label(node: Any) -> str:
    if hasattr(node, "pretty") and callable(node.pretty):
        return node.pretty()
    return str(node)


def _is_binary(node: Any) -> bool:
    return hasattr(node, "left") and hasattr(node, "right")


def _build_graph(node: Any) -> Tuple[nx.DiGraph, Dict[int, str], Dict[int, Tuple[str, ...]]]:
    G = nx.DiGraph()
    labels: Dict[int, str] = {}
    paths: Dict[int, Tuple[str, ...]] = {}

    def rec(n: Any, path: Tuple[str, ...], parent: int | None, edge_label: str | None) -> int:
        node_id = len(G)
        G.add_node(node_id)
        labels[node_id] = _get_label(n)
        paths[node_id] = path
        if parent is not None:
            G.add_edge(parent, node_id, label=edge_label or "")
        if _is_binary(n):
            rec(n.left, path + ("L",), node_id, "L")
            rec(n.right, path + ("R",), node_id, "R")
        return node_id

    rec(node, (), None, None)
    return G, labels, paths


def _tree_layout(G: nx.DiGraph, root: int = 0) -> Dict[int, Tuple[float, float]]:
    pos: Dict[int, Tuple[float, float]] = {}
    next_x = [0.0]

    def dfs(u: int, depth: int) -> None:
        children = list(G.successors(u))
        if not children:
            pos[u] = (next_x[0], -depth)
            next_x[0] += 1.0
            return
        for c in children:
            dfs(c, depth + 1)
        xs = [pos[c][0] for c in children]
        pos[u] = (sum(xs) / len(xs), -depth)

    dfs(root, 0)
    return pos


def render_kernel_graph(
    kernel: Any,
    *,
    highlight_path: Tuple[str, ...] | None = None,
    score: float | None = None,
    title: str | None = None,
    save_path: str | None = None,
    show: bool = False,
) -> None:
    G, labels, paths = _build_graph(kernel)
    pos = _tree_layout(G, root=0)

    colors = []
    for node_id in G.nodes:
        if highlight_path is not None and paths[node_id] == highlight_path:
            colors.append("#ffcc66")
        else:
            colors.append("#cfe2f3")

    plt.figure(figsize=(10, 6))
    nx.draw(
        G,
        pos,
        labels=labels,
        node_color=colors,
        node_size=900,
        font_size=8,
        arrows=False,
    )
    edge_labels = {(u, v): d.get("label", "") for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    if title or score is not None:
        score_txt = f" | score={score:.3f}" if score is not None else ""
        plt.title(f"{title or 'Kernel'}{score_txt}")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()
