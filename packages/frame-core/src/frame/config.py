"""Configuration management for frame-core."""

import os
import shutil
from pathlib import Path
from typing import Optional
try:
    import tomllib as toml  # Python 3.11+
except ImportError:  # pragma: no cover
    import tomli as toml  # type: ignore


DEFAULT_DATA_ROOT = "~/frame_data"
DEFAULT_CONFIG_TEXT = """# FRAME Configuration
# This file configures the core data management system.
# Default location: ~/.frame.toml

[frame]
# Root directory for all FRAME data (libraries and experiments)
data_root = "~/frame_data"

[frame.paths]
# Specific paths for libraries and experiments
# Variables like {data_root} will be expanded
libraries = "{data_root}/libraries"
experiments = "{data_root}/experiments"

[frame.defaults]
# Automatically migrate old data format when accessed
auto_migrate = false

# Strict validation - error if required metadata is missing
strict_validation = true
"""


def default_config_path() -> Path:
    """Return the default user config path."""
    return Path.home() / ".frame.toml"


def install_config(
    destination: Optional[Path] = None,
    overwrite: bool = False,
    source_path: Optional[Path] = None,
) -> tuple[Path, bool]:
    """Install a config file to the user location."""
    target = (destination or default_config_path()).expanduser()
    if target.exists() and not overwrite:
        return target, False

    target.parent.mkdir(parents=True, exist_ok=True)
    if source_path:
        shutil.copyfile(source_path, target)
    else:
        target.write_text(DEFAULT_CONFIG_TEXT, encoding="utf-8")
    return target, True


class FrameConfig:
    """Configuration for frame-core data management."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration.

        Args:
            config_path: Optional path to a config file.
        """
        self._config = self._load_config(config_path)
        
    def _load_config(self, config_path: Optional[Path]) -> dict:
        """Load configuration from file or use defaults."""
        if config_path is not None:
            config_path = Path(config_path).expanduser()
        defaults = {
            "frame": {
                "data_root": DEFAULT_DATA_ROOT,
                "paths": {
                    "libraries": "{data_root}/libraries",
                    "experiments": "{data_root}/experiments"
                },
                "defaults": {
                    "auto_migrate": True,
                    "strict_validation": True
                }
            }
        }

        if config_path is None:
            user_config = default_config_path()
            if user_config.exists():
                config_path = user_config
            else:
                legacy_config = Path.home() / ".frame" / "config.toml"
                workspace_config = Path.cwd() / "config" / "config.toml"
                source_config = None
                if legacy_config.exists():
                    source_config = legacy_config
                elif workspace_config.exists():
                    source_config = workspace_config
                config_path, _ = install_config(
                    destination=user_config,
                    source_path=source_config,
                )
        
        if config_path and Path(config_path).exists():
            with open(config_path, "rb") as f:
                loaded = toml.load(f)
                # Merge with defaults
                for key in defaults:
                    if key in loaded:
                        defaults[key].update(loaded[key])
        
        return defaults
    
    @property
    def data_root(self) -> Path:
        """Get the data root directory."""
        root = self._config["frame"]["data_root"]
        # Expand environment variables
        root = os.path.expandvars(root)
        return Path(root).expanduser().resolve()
    
    @property
    def libraries_path(self) -> Path:
        """Get the libraries directory."""
        path = self._config["frame"]["paths"]["libraries"]
        path = path.replace("{data_root}", str(self.data_root))
        return Path(path).expanduser().resolve()
    
    @property
    def experiments_path(self) -> Path:
        """Get the experiments directory."""
        path = self._config["frame"]["paths"]["experiments"]
        path = path.replace("{data_root}", str(self.data_root))
        return Path(path).expanduser().resolve()
    
    @property
    def auto_migrate(self) -> bool:
        """Whether to automatically migrate old data."""
        return self._config["frame"]["defaults"]["auto_migrate"]
    
    @property
    def strict_validation(self) -> bool:
        """Whether to use strict validation."""
        return self._config["frame"]["defaults"]["strict_validation"]
    
    def ensure_directories(self):
        """Create data directories if they don't exist."""
        self.libraries_path.mkdir(parents=True, exist_ok=True)
        self.experiments_path.mkdir(parents=True, exist_ok=True)


# Global configuration instance
_global_config: Optional[FrameConfig] = None


def get_config() -> FrameConfig:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = FrameConfig()
    return _global_config


def set_config(config: FrameConfig):
    """Set the global configuration instance."""
    global _global_config
    _global_config = config
