#!/usr/bin/env python3
"""从LIBERO的BDDL文件中提取场景物体信息并生成suite->task->objects映射。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import pathlib
from typing import Dict, List, Sequence


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_BDDL_ROOT = REPO_ROOT / "openpi" / "third_party" / "libero" / "libero" / "libero" / "bddl_files"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "libero" / "bddl_objects.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bddl-root",
        type=pathlib.Path,
        default=DEFAULT_BDDL_ROOT,
        help="BDDL文件所在的根目录，默认为LIBERO的bddl_files目录",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT_PATH,
        help="输出JSON文件路径，默认写入data/libero/bddl_objects.json",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON缩进，默认2",
    )
    parser.add_argument(
        "--suites",
        nargs="*",
        default=None,
        help="可选，仅处理指定的suite名称（例如 libero_spatial libero_object）",
    )
    return parser.parse_args()


def _extract_block(text: str, keyword: str) -> str | None:
    """返回关键块（如:objects）对应的字符串表示。"""
    pattern = f"(:{keyword}"
    start = text.find(pattern)
    if start == -1:
        return None

    depth = 0
    for idx in range(start, len(text)):
        char = text[idx]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _parse_names_from_block(block: str) -> List[str]:
    """解析块内部的实体名称（'-'号左侧的名称列表）。"""
    names: List[str] = []
    lines = block.splitlines()
    if len(lines) <= 2:
        return names

    for line in lines[1:-1]:
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue
        # 去掉多余括号
        cleaned = stripped.replace("(", " ").replace(")", " ")
        left = cleaned.split("-", maxsplit=1)[0]
        for token in left.split():
            if token:
                names.append(token)
    return names


def parse_bddl_objects(bddl_path: pathlib.Path) -> Sequence[str]:
    """提取单个BDDL任务中的所有场景物体名称。"""
    text = bddl_path.read_text()
    collected = []
    for section in ("fixtures", "objects"):
        block = _extract_block(text, section)
        if block:
            collected.extend(_parse_names_from_block(block))
    return sorted(dict.fromkeys(collected))


def load_suite_task_map(selected_suites: Sequence[str] | None) -> Dict[str, Sequence[str]]:
    map_path = (
        REPO_ROOT
        / "openpi"
        / "third_party"
        / "libero"
        / "libero"
        / "libero"
        / "benchmark"
        / "libero_suite_task_map.py"
    )
    spec = importlib.util.spec_from_file_location("libero_suite_task_map_local", map_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载suite映射文件: {map_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[misc]
    libero_task_map = getattr(module, "libero_task_map")

    if not selected_suites:
        return dict(libero_task_map)

    filtered = {}
    for suite in selected_suites:
        if suite not in libero_task_map:
            raise KeyError(f"未知的suite: {suite}")
        filtered[suite] = libero_task_map[suite]
    return filtered


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    suite_task_map = load_suite_task_map(args.suites)
    bddl_root = args.bddl_root.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Dict[str, Sequence[str]]] = {}
    for suite, tasks in suite_task_map.items():
        logging.info("处理suite: %s", suite)
        suite_result: Dict[str, Sequence[str]] = {}
        for task_name in tasks:
            bddl_file = bddl_root / suite / f"{task_name}.bddl"
            if not bddl_file.exists():
                raise FileNotFoundError(f"找不到BDDL文件: {bddl_file}")
            suite_result[task_name] = parse_bddl_objects(bddl_file)
        result[suite] = suite_result

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=args.indent)
        f.write("\n")

    logging.info("完成！共写入%s", output_path)


if __name__ == "__main__":
    main()

