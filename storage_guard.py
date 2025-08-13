
#!/usr/bin/env python3
import os, sys, time, argparse
from pathlib import Path

def dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except Exception:
                pass
    return total

def collect_files(path: Path):
    files = []
    for p in path.rglob("*"):
        if p.is_file():
            try:
                files.append((p, p.stat().st_mtime, p.stat().st_size))
            except Exception:
                pass
    # sort oldest first
    files.sort(key=lambda x: x[1])
    return files

def prune_by_age_and_size(root: Path, max_age_hours: float, max_total_gb: float, dry_run: bool=False):
    now = time.time()
    max_total_bytes = int(max_total_gb * (1024**3))
    files = collect_files(root)

    # 1) Remove files older than max_age_hours
    removed = 0
    for p, mtime, size in list(files):
        age_hours = (now - mtime) / 3600.0
        if age_hours > max_age_hours:
            if not dry_run:
                try:
                    p.unlink()
                    removed += 1
                except Exception as e:
                    print(f"[warn] failed to delete {p}: {e}")
            files.remove((p, mtime, size))

    # 2) Enforce global cap (oldest-first)
    total = sum(size for _,_,size in files)
    i = 0
    while total > max_total_bytes and i < len(files):
        p, mtime, size = files[i]
        if not dry_run:
            try:
                p.unlink()
            except Exception as e:
                print(f"[warn] failed to delete {p}: {e}")
        total -= size
        i += 1
        removed += 1

    return removed

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Prune output directory by age and global size cap.")
    ap.add_argument("--root", default="output", help="Root folder to prune (default: output)")
    ap.add_argument("--max-age-hours", type=float, default=6.0, help="Delete anything older than this many hours.")
    ap.add_argument("--max-total-gb", type=float, default=3.0, help="Keep total under this cap (GB).")
    ap.add_argument("--dry-run", action="store_true", help="Only print what would be deleted.")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[storage_guard] nothing to do, {root} does not exist")
        sys.exit(0)

    removed = prune_by_age_and_size(root, args.max_age_hours, args.max_total_gb, args.dry_run)
    print(f"[storage_guard] removed {removed} files; done.")
