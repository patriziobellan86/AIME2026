from __future__ import annotations

import json
from pathlib import Path

# =======================
# CONFIG: cambia solo qui
# =======================
ROOT_DIR = "root-dir (the path to Experiment folder of this repo"
OLD = "./Experiments"
NEW = "Fixed folder, e.g., the absolute path to Experiment folder"

DRY_RUN = False        # True = do not change files, just print out the changes
MAKE_BACKUP = False
VALIDATE_JSON = True


def safe_read_text(fp: Path) -> str:
    try:
        return fp.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return fp.read_text(encoding="utf-8", errors="replace")


def safe_write_text(fp: Path, text: str) -> None:
    fp.write_text(text, encoding="utf-8")


def replace_in_file(fp: Path) -> int:
    """Returns number of replacements performed (0 if none)."""
    text = safe_read_text(fp)
    if OLD not in text:
        return 0

    new_text = text.replace(OLD, NEW)
    n_rep = text.count(OLD)

    if fp.suffix.lower() == ".json" and VALIDATE_JSON:
        try:
            json.loads(new_text)
        except Exception as e:
            print(f"[WARN] JSON non valido dopo la sostituzione: {fp} -> {e}")
            # Se vuoi BLOCCARE in caso di errore, decommenta:
            # return 0

    print(f"[CHANGE] {fp}  (replacements: {n_rep})")

    if not DRY_RUN:
        if MAKE_BACKUP:
            bak = fp.with_suffix(fp.suffix + ".bak")
            if not bak.exists():
                safe_write_text(bak, text)
        safe_write_text(fp, new_text)

    return n_rep


def main() -> None:
    root = Path(ROOT_DIR).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"ROOT_DIR non trovato: {root}")

    changed_files = 0
    total_replacements = 0

    # Cerca solo .json e .py, ricorsivamente
    for fp in root.rglob("*"):
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in {".json", ".py"}:
            continue

        n = replace_in_file(fp)
        if n > 0:
            changed_files += 1
            total_replacements += n

    print(f"\nDone. Files changed: {changed_files} | Total replacements: {total_replacements}")
    if DRY_RUN:
        print("DRY_RUN=True: nessun file è stato modificato.")
    elif MAKE_BACKUP:
        print("Backup creati con estensione .bak (se non già presenti).")


if __name__ == "__main__":
    main()
