"""Utility functions for management layer."""

import os
import stat
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def generate_uuid(prefix: str = "") -> str:
    """Generate a UUID with optional prefix.
    
    Args:
        prefix: Prefix for the UUID (e.g., 'lib', 'exp', 'ckpt')
    
    Returns:
        UUID string
    """
    uid = uuid.uuid4().hex[:12]
    if prefix:
        return f"{prefix}_{uid}"
    return uid


def timestamp_iso() -> str:
    """Generate ISO 8601 timestamp in UTC."""
    return datetime.now(timezone.utc).isoformat()


def write_protect(path: Path):
    """Make file read-only for owner, group, and others.
    
    Args:
        path: Path to file to write-protect
    """
    if not path.exists():
        return
    
    if path.is_file():
        os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    elif path.is_dir():
        # Make directory executable (readable + searchable)
        os.chmod(path, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)


def write_protect_tree(directory: Path, exclude_patterns: Optional[list[str]] = None):
    """Recursively write-protect all files except patterns.
    
    Args:
        directory: Root directory to protect
        exclude_patterns: Patterns to exclude (e.g., ['logs/', 'tensorboard/'])
    """
    exclude_patterns = exclude_patterns or ['logs/', 'tensorboard/']
    
    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        if any(pattern in str(root) for pattern in exclude_patterns):
            continue
        
        # Protect directory itself
        write_protect(Path(root))
        
        # Protect all files
        for file in files:
            file_path = Path(root) / file
            write_protect(file_path)


def is_uuid(s: str) -> bool:
    """Check if string looks like a UUID (with or without prefix).
    
    Args:
        s: String to check
    
    Returns:
        True if string looks like a UUID
    """
    # Check for prefix_xxx format
    if "_" in s:
        prefix, rest = s.split("_", 1)
        if len(rest) == 12 and all(c in "0123456789abcdef" for c in rest):
            return True
    
    # Check for plain hex string
    if len(s) == 12 and all(c in "0123456789abcdef" for c in s):
        return True
    
    # Check for full UUID format
    try:
        uuid.UUID(s)
        return True
    except ValueError:
        return False


def resolve_path_or_uuid(path_or_uuid: str, search_paths: list[Path]) -> Optional[Path]:
    """Resolve a path or UUID to an actual path.
    
    Args:
        path_or_uuid: Either a filesystem path or a UUID
        search_paths: Directories to search for UUIDs
    
    Returns:
        Resolved path or None if not found
    """
    # If it's a regular path, just return it
    path = Path(path_or_uuid)
    if path.exists():
        return path.resolve()
    
    # If it looks like a UUID, search for it
    if is_uuid(path_or_uuid):
        for search_path in search_paths:
            candidate = search_path / path_or_uuid
            if candidate.exists():
                return candidate.resolve()
    
    return None

