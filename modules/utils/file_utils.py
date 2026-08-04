from pathlib import Path


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_uploaded_file(uploaded_file, target_dir: Path) -> Path:
    """Save a Streamlit uploaded file object into the target directory."""
    ensure_directory(target_dir)
    target_path = target_dir / uploaded_file.name
    target_path.write_bytes(uploaded_file.getbuffer())
    return target_path