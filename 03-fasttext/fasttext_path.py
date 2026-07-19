import hashlib
import os
import shutil
from pathlib import Path


def _is_ascii_path(path):
    try:
        str(path).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def fasttext_read_path(path):
    if _is_ascii_path(path):
        return str(path)

    source = Path(path)
    temp_dir = Path(os.environ.get("SystemDrive", "C:") + "\\fasttext_tmp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:12]
    target = temp_dir / f"{digest}_{source.name}"
    if not target.exists() or target.stat().st_mtime < source.stat().st_mtime:
        shutil.copyfile(source, target)
    return str(target)


def fasttext_write_path(path):
    if _is_ascii_path(path):
        return str(path), None

    target = Path(path)
    temp_dir = Path(os.environ.get("SystemDrive", "C:") + "\\fasttext_tmp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(str(target).encode("utf-8")).hexdigest()[:12]
    temp_path = temp_dir / f"{digest}_{target.name}"
    return str(temp_path), str(target)


def copy_back(temp_path, target_path):
    if target_path:
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(temp_path, target_path)
