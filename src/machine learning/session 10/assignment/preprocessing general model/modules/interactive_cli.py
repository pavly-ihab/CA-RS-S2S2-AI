# modules/interactive_cli.py
"""
Universal Interactive CLI Helper
Provides rich terminal menus, prompts, column pickers, and decision gates
with validation, clear defaults, and support for automated non-interactive runs.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional


def print_step_header(step_num: int, step_name: str) -> None:
    """Prints a styled banner for a preprocessing step."""
    banner = f" STEP {step_num}: {step_name.upper()} "
    print("\n" + "=" * 70)
    print(f"{banner.center(70, '=')}")
    print("=" * 70)


def prompt_user_choice(
    prompt_text: str,
    options: Dict[str, str],
    default_key: str,
    auto_mode: bool = False
) -> str:
    """
    Displays a multiple-choice menu to the user and returns the chosen key.
    """
    print(f"\n[?] {prompt_text}")
    for key, desc in options.items():
        is_default = " (Default)" if key == default_key else ""
        print(f"    [{key}] {desc}{is_default}")

    if auto_mode:
        print(f"    -> Auto-selected default: [{default_key}] {options.get(default_key)}")
        return default_key

    while True:
        try:
            choice = input(f"Select option [{default_key}]: ").strip()
            if not choice:
                return default_key
            if choice in options:
                return choice
            matched = [k for k in options if k.lower() == choice.lower()]
            if matched:
                return matched[0]
            print(f"[!] Invalid selection '{choice}'. Please choose from: {list(options.keys())}")
        except (KeyboardInterrupt, EOFError):
            print(f"\n[!] Input cancelled. Using default: {default_key}")
            return default_key


def prompt_float(
    prompt_text: str,
    default_val: float,
    min_val: float = 0.0,
    max_val: float = 1.0,
    auto_mode: bool = False
) -> float:
    """Prompts the user for a floating-point number within a range."""
    if auto_mode:
        print(f"[?] {prompt_text} -> Auto-selected: {default_val}")
        return default_val

    while True:
        try:
            val_str = input(f"[?] {prompt_text} [{default_val}]: ").strip()
            if not val_str:
                return default_val
            val = float(val_str)
            if min_val <= val <= max_val:
                return val
            print(f"[!] Value must be between {min_val} and {max_val}.")
        except ValueError:
            print("[!] Please enter a valid numerical value.")
        except (KeyboardInterrupt, EOFError):
            print(f"\n[!] Input cancelled. Using default: {default_val}")
            return default_val


def prompt_dataset_path(default_path: Path, auto_mode: bool = False) -> Path:
    """Prompts the user for a tabular dataset file path."""
    if auto_mode:
        return default_path

    while True:
        try:
            raw_in = input(f"[?] Enter dataset path [{default_path}]: ").strip()
            if not raw_in:
                return default_path
            path_obj = Path(raw_in).expanduser().resolve()
            if path_obj.exists() and path_obj.is_file():
                return path_obj
            print(f"[!] File not found at '{path_obj}'. Please provide a valid file path.")
        except (KeyboardInterrupt, EOFError):
            print(f"\n[!] Using default dataset: {default_path}")
            return default_path


def prompt_column_selection(
    prompt_text: str,
    columns: List[str],
    default_col: Optional[str] = None,
    auto_mode: bool = False
) -> str:
    """Allows selecting a single column from a numbered list."""
    if not columns:
        raise ValueError("No columns provided to select from.")

    default_col = default_col if (default_col and default_col in columns) else columns[-1]

    if auto_mode:
        print(f"[?] {prompt_text} -> Auto-selected: '{default_col}'")
        return default_col

    print(f"\n[?] {prompt_text}")
    for idx, col in enumerate(columns, 1):
        is_def = " (Default)" if col == default_col else ""
        print(f"    [{idx}] {col}{is_def}")

    while True:
        try:
            choice = input(f"Enter column number or name [{default_col}]: ").strip()
            if not choice:
                return default_col
            if choice in columns:
                return choice
            if choice.isdigit():
                num = int(choice)
                if 1 <= num <= len(columns):
                    return columns[num - 1]
            print(f"[!] Invalid choice. Enter a number 1-{len(columns)} or exact column name.")
        except (KeyboardInterrupt, EOFError):
            print(f"\n[!] Using default: '{default_col}'")
            return default_col


def prompt_multi_column_selection(
    prompt_text: str,
    columns: List[str],
    suggested_cols: Optional[List[str]] = None,
    auto_mode: bool = False
) -> List[str]:
    """Allows selecting multiple columns by numbers or names, or 'none'."""
    suggested_cols = [c for c in (suggested_cols or []) if c in columns]

    if auto_mode:
        print(f"[?] {prompt_text} -> Auto-dropped: {suggested_cols}")
        return suggested_cols

    print(f"\n[?] {prompt_text}")
    print("    Available columns:")
    for idx, col in enumerate(columns, 1):
        is_sug = " [SUGGESTED FOR REMOVAL]" if col in suggested_cols else ""
        print(f"    [{idx}] {col}{is_sug}")
    print("    [none] Do not drop any columns")

    default_str = ", ".join([str(columns.index(c) + 1) for c in suggested_cols]) if suggested_cols else "none"

    while True:
        try:
            choice = input(f"Enter numbers or names separated by commas [{default_str}]: ").strip()
            if not choice:
                return suggested_cols
            if choice.lower() in ["none", "no", "0"]:
                return []

            selected = []
            tokens = [t.strip() for t in choice.split(",") if t.strip()]
            valid = True
            for token in tokens:
                if token in columns:
                    selected.append(token)
                elif token.isdigit() and 1 <= int(token) <= len(columns):
                    selected.append(columns[int(token) - 1])
                else:
                    print(f"[!] Unrecognized column token: '{token}'")
                    valid = False
                    break
            if valid:
                return list(dict.fromkeys(selected))
        except (KeyboardInterrupt, EOFError):
            print(f"\n[!] Using suggested: {suggested_cols}")
            return suggested_cols
