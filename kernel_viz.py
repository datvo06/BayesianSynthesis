from __future__ import annotations

from typing import Any, Dict, Tuple

import re
import os

from graphviz import Digraph


def _split_name_token(token: str) -> tuple[str, str | None]:
    m = re.match(r"([A-Za-z_]+)\(([^)]+)\)", token)
    if m:
        return m.group(1), m.group(2)
    return token, None


def _shorten_id(text: str | None) -> str | None:
    if not text:
        return text
    m = re.search(r"(\d+)$", text)
    if m:
        return text[: max(0, len(text) - len(m.group(1)))] + m.group(1)[-2:]
    return text


def _pretty_kernel_token(token: str) -> str:
    token = token.strip()
    if token.lower() in {"none", "null", "nil"}:
        return "X"
    base = re.sub(r"\d+$", "", token).lower()
    mapping = {
        "rbf": "RBF",
        "per": "Per",
        "periodic": "Per",
        "lin": "Lin",
        "linear": "Lin",
        "wn": "WN",
        "white": "WN",
    }
    if base in mapping:
        return mapping[base]
    return re.sub(r"\d+$", "", token)


def _get_label(node: Any) -> str:
    if hasattr(node, "op") and getattr(node, "op") is not None:
        return str(getattr(node, "op"))
    if node.__class__.__name__ == "KernelNode":
        name = getattr(node, "name", None)
        if not name or str(name).lower() in {"none", "null", "nil"}:
            return "X"
        base, _detail = _split_name_token(str(name))
        return _pretty_kernel_token(base)
    if hasattr(node, "name"):
        type_name = node.__class__.__name__
        name = _shorten_id(getattr(node, "name", None))
        if name:
            return f"{type_name}\n{_pretty_kernel_token(name)}"
        return type_name
    if node.__class__.__name__ == "Sum":
        return "+"
    if node.__class__.__name__ == "Product":
        return "*"
    if hasattr(node, "pretty") and callable(node.pretty):
        text = node.pretty()
        base, detail = _split_name_token(text)
        detail = _shorten_id(detail)
        base = _pretty_kernel_token(base)
        if detail:
            return f"{base}\n{_pretty_kernel_token(detail)}"
        return base
    text = str(node)
    base, detail = _split_name_token(text)
    detail = _shorten_id(detail)
    base = _pretty_kernel_token(base)
    if detail:
        return f"{base}\n{_pretty_kernel_token(detail)}"
    return base


def _is_binary(node: Any) -> bool:
    return hasattr(node, "left") and hasattr(node, "right")


def _build_graph(
    node: Any,
    *,
    base_fillcolor: str = "#ffffff",
) -> Tuple[Digraph, Dict[int, Tuple[str, ...]], Dict[int, str]]:
    graph = Digraph("kernel")
    graph.attr("node", shape="box", style="rounded,filled", fontname="Helvetica")
    paths: Dict[int, Tuple[str, ...]] = {}
    labels: Dict[int, str] = {}

    def rec(n: Any, path: Tuple[str, ...], parent: int | None, edge_label: str | None) -> int:
        node_id = len(paths)
        paths[node_id] = path
        labels[node_id] = _get_label(n)
        graph.node(str(node_id), labels[node_id], fillcolor=base_fillcolor)
        if parent is not None:
            graph.edge(str(parent), str(node_id), label=edge_label or "")
        if _is_binary(n):
            rec(n.left, path + ("L",), node_id, "L")
            rec(n.right, path + ("R",), node_id, "R")
        return node_id

    rec(node, (), None, None)
    return graph, paths, labels


def _add_kernel_to_graph(
    graph: Digraph,
    node: Any,
    prefix: str,
    highlight_paths: Dict[Tuple[str, ...], str] | None,
    *,
    base_fillcolor: str = "#ffffff",
) -> Tuple[str, Dict[str, Tuple[str, ...]], Dict[str, str]]:
    paths: Dict[str, Tuple[str, ...]] = {}
    labels: Dict[str, str] = {}

    def rec(n: Any, path: Tuple[str, ...], parent: str | None, edge_label: str | None) -> str:
        node_id = f"{prefix}_{len(paths)}"
        paths[node_id] = path
        labels[node_id] = _get_label(n)
        fill = base_fillcolor
        if highlight_paths:
            fill = highlight_paths.get(path, base_fillcolor)
        graph.node(node_id, labels[node_id], fillcolor=fill)
        if parent is not None:
            graph.edge(parent, node_id, label=edge_label or "")
        if _is_binary(n):
            rec(n.left, path + ("L",), node_id, "L")
            rec(n.right, path + ("R",), node_id, "R")
        return node_id

    root_id = rec(node, (), None, None)
    return root_id, paths, labels


def render_kernel_graph(
    kernel: Any,
    *,
    highlight_path: Tuple[str, ...] | None = None,
    highlight_color: str = "#ffcc66",
    highlight_paths: Dict[Tuple[str, ...], str] | None = None,
    base_fillcolor: str = "#ffffff",
    score: float | None = None,
    title: str | None = None,
    save_path: str | None = None,
) -> None:
    graph, paths, labels = _build_graph(kernel, base_fillcolor=base_fillcolor)

    highlight_map: Dict[Tuple[str, ...], str] = {}
    if highlight_paths:
        highlight_map.update(highlight_paths)
    elif highlight_path is not None:
        highlight_map[highlight_path] = highlight_color

    if highlight_map:
        for node_id, path in paths.items():
            color = highlight_map.get(path)
            if color:
                graph.node(str(node_id), labels[node_id], fillcolor=color)

    if title or score is not None:
        score_txt = f" | score={score:.3f}" if score is not None else ""
        graph.attr(label=f"{title or 'Kernel'}{score_txt}", labelloc="t")

    if save_path:
        base, _ = os.path.splitext(save_path)
        graph.render(base, format="png", cleanup=True)


def render_kernel_chain(
    kernels: list[Any],
    scores: list[float | None],
    *,
    title: str,
    save_path: str,
) -> None:
    graph = Digraph("kernel_chain")
    graph.attr("node", shape="box", style="rounded,filled", fontname="Helvetica")
    graph.attr(rankdir="LR")

    roots: list[str] = []
    for idx, k in enumerate(kernels):
        prefix = f"k{idx}"
        with graph.subgraph(name=f"cluster_{prefix}") as sub:
            sub.attr(rankdir="TB")
            root_id, _, _ = _add_kernel_to_graph(sub, k, prefix, None, base_fillcolor="#ffffff")
        score_txt = "" if scores[idx] is None else f"{scores[idx]:.3f}"
        graph.node(f"{prefix}_label", f"Step {idx}\nscore={score_txt}", shape="note", fillcolor="#f8f8f8")
        graph.edge(f"{prefix}_label", root_id, style="dashed")
        roots.append(root_id)

    with graph.subgraph(name="cluster_align") as sub:
        sub.attr(rank="same")
        for root_id in roots:
            sub.node(root_id)

    for i in range(len(roots) - 1):
        graph.edge(roots[i], roots[i + 1], style="bold", color="#666666")

    graph.attr(label=title, labelloc="t")
    base, _ = os.path.splitext(save_path)
    graph.render(base, format="png", cleanup=True)


def render_kernel_mutation(
    before: Any,
    after: Any,
    *,
    sever_path: Tuple[str, ...] | None,
    score: float | None,
    title: str,
    save_path: str,
) -> None:
    graph = Digraph("kernel_mutation")
    graph.attr("node", shape="box", style="rounded,filled", fontname="Helvetica")
    graph.attr(rankdir="LR")

    before_paths: Dict[str, Tuple[str, ...]] = {}
    after_paths: Dict[str, Tuple[str, ...]] = {}

    with graph.subgraph(name="cluster_before") as sub:
        sub.attr(label="Before", color="#dddddd")
        _, before_paths, _ = _add_kernel_to_graph(
            sub,
            before,
            "b",
            {sever_path: "#ff6666"} if sever_path is not None else None,
            base_fillcolor="#ffffff",
        )

    with graph.subgraph(name="cluster_after") as sub:
        sub.attr(label="After", color="#dddddd")
        _, after_paths, _ = _add_kernel_to_graph(
            sub,
            after,
            "a",
            {sever_path: "#66cc66"} if sever_path is not None else None,
            base_fillcolor="#ffffff",
        )

    if sever_path is not None:
        before_id = next((k for k, v in before_paths.items() if v == sever_path), None)
        after_id = next((k for k, v in after_paths.items() if v == sever_path), None)
        if before_id and after_id:
            graph.edge(before_id, after_id, style="dashed", color="#999999")

    score_txt = f" | score={score:.3f}" if score is not None else ""
    graph.attr(label=f"{title}{score_txt}", labelloc="t")
    base, _ = os.path.splitext(save_path)
    graph.render(base, format="png", cleanup=True)
