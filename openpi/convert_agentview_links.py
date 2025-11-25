from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict
from urllib.parse import urlparse


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Convert multi-view LIBERO init-frame links to single-view format "
            "that only keeps the desired camera (agentview by default)."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=root
        / "libero-init-frames_new"
        / "json_data_for_rl"
        / "vlm_initial_state_links.json",
        help="Path to the multi-view JSON produced by extract_and_upload_libero_multiview_initframes_hf.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root
        / "libero-init-frames_new"
        / "json_data_for_rl"
        / "vlm_initial_state_links_new.json",
        help="Destination JSON file (will be overwritten).",
    )
    parser.add_argument(
        "--view-key",
        default="agentview",
        help="Camera key to keep (matches keys inside the input JSON).",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=root / "libero-init-frames_new",
        help="Directory that stores the downloaded init-frame images.",
    )
    parser.add_argument(
        "--relative-to",
        type=Path,
        default=None,
        help=(
            "Base directory for computing relative paths in the output JSON. "
            "Defaults to the current working directory at runtime."
        ),
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=4,
        help="JSON indentation level for the output file.",
    )
    return parser.parse_args()


def select_view(entry: Any, view_key: str) -> str:
    if isinstance(entry, dict):
        if view_key in entry:
            return entry[view_key]
        for key, value in entry.items():
            if view_key in key:
                return value
        raise KeyError(f"View '{view_key}' not found in {entry.keys()}")
    if isinstance(entry, list):
        for value in entry:
            if isinstance(value, str) and view_key in value:
                return value
        if entry:
            raise KeyError(
                f"List entry provided but no element contains view '{view_key}'"
            )
        raise KeyError("Empty list provided for view selection.")
    if isinstance(entry, str):
        if view_key in entry:
            return entry
        raise KeyError(
            f"String entry does not contain the requested view '{view_key}'."
        )
    raise TypeError(f"Unsupported entry type {type(entry)}; expected dict/list/str.")


def locate_image(link: str, images_dir: Path) -> Path:
    """Resolve the local path for a given link, validating the file exists."""
    images_dir = images_dir.resolve()
    if "://" in link:
        parsed = urlparse(link)
        candidate = images_dir / Path(parsed.path).name
    else:
        candidate_path = Path(link)
        candidate = (
            candidate_path
            if candidate_path.is_absolute()
            else images_dir / candidate_path.name
        )
    candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(
            f"Could not find local file for link '{link}' at {candidate}"
        )
    return candidate


def make_link_converter(
    view_key: str, images_dir: Path, relative_to: Path
) -> Callable[[Any], str]:
    base_dir = relative_to.resolve()

    def _convert(entry: Any) -> str:
        link = select_view(entry, view_key)
        local_path = locate_image(link, images_dir)
        try:
            rel_path = local_path.relative_to(base_dir)
        except ValueError:
            rel_path = Path(os.path.relpath(local_path, base_dir))
        return rel_path.as_posix()

    return _convert


def convert_links(
    source: Dict[str, Dict[str, Any]], converter: Callable[[Any], str]
) -> Dict[str, Dict[str, str]]:
    converted: Dict[str, Dict[str, str]] = {}
    for benchmark, tasks in source.items():
        suite_result: Dict[str, str] = {}
        for task_name, entry in tasks.items():
            suite_result[task_name] = converter(entry)
        converted[benchmark] = suite_result
    return converted


def main() -> None:
    args = parse_args()
    if args.relative_to is None:
        args.relative_to = Path.cwd()
    with args.input.open("r", encoding="utf-8") as infile:
        source_data = json.load(infile)

    converter = make_link_converter(args.view_key, args.images_dir, args.relative_to)
    converted = convert_links(source_data, converter)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as outfile:
        json.dump(converted, outfile, indent=args.indent, ensure_ascii=False)
        outfile.write("\n")

    task_count = sum(len(tasks) for tasks in converted.values())
    print(
        f"Wrote {task_count} tasks to {args.output} "
        f"using view '{args.view_key}'."
    )


if __name__ == "__main__":
    main()

