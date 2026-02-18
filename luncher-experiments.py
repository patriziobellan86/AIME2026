#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path
from typing import List, Set, Tuple
import os

from PoE3.PoE.FileToolkit import get_outputdir

# ----------------------------
# Helpers
# ----------------------------
def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_baseline_config(config_path: Path) -> bool:
    try:
        data = read_json(config_path)
        return bool(data.get("baseline", False))
    except Exception:
        # if the file is corrupted/unreadable, treat it as non-baseline to avoid blocking the listing
        return False


def parse_indices(selection: str, max_n: int) -> List[int]:
    """
    Accepts: "all" or "1,2,5-7" (1-based indices).
    """
    s = selection.strip().lower()
    if s == "all":
        return list(range(1, max_n + 1))

    out: Set[int] = set()
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            a, b = a.strip(), b.strip()
            if not a.isdigit() or not b.isdigit():
                raise ValueError(f"Invalid selection: '{chunk}'")
            ia, ib = int(a), int(b)
            if ia > ib:
                ia, ib = ib, ia
            out.update(range(ia, ib + 1))
        else:
            if not chunk.isdigit():
                raise ValueError(f"Invalid selection: '{chunk}'")
            out.add(int(chunk))

    bad = [x for x in out if x < 1 or x > max_n]
    if bad:
        raise ValueError(f"Indices out of range: {bad} (valid range: 1..{max_n})")

    return sorted(out)


def pick_one(title: str, options: List[str]) -> str:
    if not options:
        raise ValueError(f"No available options: {title}")

    print(f"\n{title}")
    for i, opt in enumerate(options, start=1):
        print(f"  [{i:>3}] {opt}")

    while True:
        raw = input("Select a number: ").strip()
        if not raw.isdigit():
            print("Invalid input.")
            continue
        idx = int(raw)
        if 1 <= idx <= len(options):
            return options[idx - 1]
        print("Index out of range.")


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = " [Y/n] " if default else " [y/N] "
    while True:
        raw = input(prompt + suffix).strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("Invalid answer (y/n).")


def choose_actions() -> List[str]:
    print("\nChoose what to run:")
    print("  [1] initialization")
    print("  [2] experiments")
    print("  [3] final_decision_maker")
    print("  [4] initialization + experiments")
    print("  [5] experiments + final_decision_maker")
    print("  [6] all (initialization + experiments + final_decision_maker)")
    print("  [0] quit")

    while True:
        raw = input("Select a number: ").strip()
        if not raw.isdigit():
            print("Invalid input.")
            continue
        v = int(raw)
        if v == 0:
            return []
        if v == 1:
            return ["initialization"]
        if v == 2:
            return ["experiments"]
        if v == 3:
            return ["final_decision_maker"]
        if v == 4:
            return ["initialization", "experiments"]
        if v == 5:
            return ["experiments", "final_decision_maker"]
        if v == 6:
            return ["initialization", "experiments", "final_decision_maker"]
        print("Index out of range.")


# ----------------------------
# Discover folders/files
# ----------------------------
def config_root(project_root: Path) -> Path:
    return (project_root / "Config_files" / "experiments_config_files").resolve()


def list_dataset_types(cfg_root: Path) -> List[str]:
    if not cfg_root.exists():
        return []
    return sorted([p.name for p in cfg_root.iterdir() if p.is_dir()])


def list_dataset_rel_paths(cfg_root: Path, dataset_type: str) -> List[str]:
    """
    Returns all subfolders (including nested ones) that contain at least one config_file_*.json
    under: cfg_root/dataset_type/
    """
    base = (cfg_root / dataset_type).resolve()
    if not base.exists():
        return []

    rels: Set[str] = set()
    for d in base.rglob("*"):
        if not d.is_dir():
            continue
        matches = list(d.glob("config_file_*.json"))
        if matches:
            rels.add(str(d.relative_to(base)).replace("\\", "/"))

    # if configs are directly under base (no nesting), include "."
    direct = list(base.glob("config_file_*.json"))
    if direct:
        rels.add(".")

    # sort putting "." first if present
    rel_list = sorted(rels)
    if "." in rel_list:
        rel_list.remove(".")
        rel_list.insert(0, ".")
    return rel_list


def completion_percentage(config_file: Path) -> int:
    # read args_dict
    args_dict = json.load(open(config_file, "r"))
    output_dir = get_outputdir(args_dict)

    # check input size
    input_file = args_dict["input"]
    if input_file.endswith(".json"):
        input_size = len(json.load(open(input_file, "r")))
    elif input_file.endswith(".txt"):
        input_size = len([x.strip() for x in open(input_file, "r")])

    # check size of query answers
    answers_file = Path(output_dir).joinpath("queries_answers.json").resolve().absolute().__str__()
    if not os.path.exists(answers_file):
        return 0
    answers = json.load(open(answers_file, "r"))
    answers_size = len(answers)

    return round(answers_size / input_size, 2) * 100


def list_config_files(cfg_root: Path, dataset_type: str, dataset_rel: str) -> List[Path]:
    base = (cfg_root / dataset_type).resolve()
    folder = base if dataset_rel == "." else (base / dataset_rel).resolve()
    pattern = (folder / "config_file_*.json").absolute().__str__()

    list_config = sorted([Path(p).resolve() for p in glob(pattern)])
    return list_config


def choose_configs(files: List[Path]) -> List[Path]:
    if not files:
        return []

    print("\nAvailable Configs:")
    for i, p in enumerate(files, start=1):
        tag = "baseline" if is_baseline_config(p) else "poe"

        perc_completed = completion_percentage(p)
        if perc_completed == 100:
            print(f"  [{'DONE':>15}] {p.name:>25}   ({tag})")
        else:
            print(f"  [{i:<3}]{perc_completed:>8}% {p.name:>25}   ({tag})")

    while True:
        raw = input("Select configs (all or e.g., 1,3,5-7): ").strip()
        try:
            idxs = parse_indices(raw, len(files))
            return [files[i - 1] for i in idxs]
        except Exception as e:
            print(f"Error: {e}")


# ----------------------------
# Runners
# ----------------------------
def run_initialization(cfg: Path) -> None:
    # baseline: do not initialize (PoE not needed)
    # if is_baseline_config(cfg):
    #     print(f"[INIT] SKIP baseline: {cfg.name}")
    #     return

    from PoE3.PoE.initialization import initialize_poe_from_config_file
    print(f"[INIT] {cfg.name}")
    initialize_poe_from_config_file(cfg.absolute())


def run_experiments(cfg: Path) -> None:
    if is_baseline_config(cfg):
        from PoE3.run_baseline_from_config_file import run_baseline_from_config_file
        print(f"[EXP] baseline: {cfg.name}")
        run_baseline_from_config_file(cfg.absolute())
    else:
        from PoE3.run_queries_to_expert_agents import run_queries_to_expert_agents_from_config_file
        print(f"[EXP] poe: {cfg.name}")
        run_queries_to_expert_agents_from_config_file(cfg.absolute())


def run_final_decision_maker(cfg: Path, include_baseline: bool) -> None:
    if is_baseline_config(cfg) and not include_baseline:
        print(f"[FDM] SKIP baseline: {cfg.name}")
        return

    from PoE3.run_queries_to_final_decision_maker_agent import run_final_decision_maker_from_config_file
    print(f"[FDM] {cfg.name}")
    run_final_decision_maker_from_config_file(cfg.absolute())


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Interactive interface to select configs and run initialization/experiments/FDM."
    )
    ap.add_argument(
        "--project-root",
        default="PATH TO FOLDER ./Experiments",
        help="Root folder for ./Experiments",
    )
    args = ap.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    cfg_root = config_root(project_root)

    if not cfg_root.exists():
        raise FileNotFoundError(f"Config folder not found: {cfg_root}")

    # 1) dataset_type
    dtypes = list_dataset_types(cfg_root)
    if not dtypes:
        raise FileNotFoundError(f"No dataset_type found under: {cfg_root}")

    dataset_type = pick_one("Select dataset_type:", dtypes)

    # 2) dataset (rel path)
    datasets = list_dataset_rel_paths(cfg_root, dataset_type)
    if not datasets:
        raise FileNotFoundError(f"No dataset with config_file_*.json under: {cfg_root / dataset_type}")

    dataset_rel = pick_one(f"Select dataset under '{dataset_type}':", datasets)

    # 3) configs
    files = list_config_files(cfg_root, dataset_type, dataset_rel)
    if not files:
        raise FileNotFoundError("No config_file_*.json found in the selected folder.")

    # optional: filter baseline/non-baseline before selection
    print("\nConfig filter:")
    print("  [1] no filter (all)")
    print("  [2] baseline only")
    print("  [3] non-baseline only (PoE)")
    while True:
        raw = input("Select a number: ").strip()
        if raw.isdigit() and int(raw) in (1, 2, 3):
            choice = int(raw)
            break
        print("Invalid input.")

    if choice == 2:
        files = [f for f in files if is_baseline_config(f)]
    elif choice == 3:
        files = [f for f in files if not is_baseline_config(f)]

    if not files:
        raise FileNotFoundError("No configs available after filtering.")

    # NEW: choose whether to run ALL configs automatically or pick manually
    print("\nRun mode:")
    print("  [1] Manually select configs")
    print("  [2] Run ALL configs (all frameworks) in the selected folder")
    while True:
        raw = input("Select a number: ").strip()
        if raw.isdigit() and int(raw) in (1, 2):
            run_mode = int(raw)
            break
        print("Invalid input.")

    if run_mode == 2:
        selected = files  # run all frameworks/configs
    else:
        selected = choose_configs(files)
        if not selected:
            return

    # 4) actions
    actions = choose_actions()
    if not actions:
        return

    include_baseline_fdm = False
    if "final_decision_maker" in actions:
        include_baseline_fdm = ask_yes_no("Run Final Decision Maker also for baselines?", default=False)

    stop_on_error = True  # ask_yes_no("Stop at the first error?", default=False)

    # 5) run
    folder = (cfg_root / dataset_type) if dataset_rel == "." else (cfg_root / dataset_type / dataset_rel)
    print("\n--- RUN SUMMARY ---")
    print(f"Project root : {project_root}")
    print(f"Config root  : {cfg_root}")
    print(f"Folder       : {folder.resolve()}")
    print(f"Configs      : {len(selected)}")
    print(f"Actions      : {actions}")
    if "final_decision_maker" in actions:
        print(f"FDM baseline : {include_baseline_fdm}")
    print("-------------------\n")

    for i, cfg in enumerate(selected, start=1):
        print(f"({i}/{len(selected)}) {cfg.name}")
        try:
            if "initialization" in actions:
                run_initialization(cfg)
            if "experiments" in actions:
                run_experiments(cfg)
            if "final_decision_maker" in actions:
                run_final_decision_maker(cfg, include_baseline=include_baseline_fdm)
        except Exception as e:
            print(f"[ERROR] {cfg.name}: {e}")
            if stop_on_error:
                raise


if __name__ == "__main__":
    main()
