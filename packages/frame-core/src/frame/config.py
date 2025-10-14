"""Configuration management for frame-core."""

import os
from pathlib import Path
from typing import Optional
try:
    import tomllib as toml  # Python 3.11+
except ImportError:  # pragma: no cover
    import tomli as toml  # type: ignore


class FrameConfig:
    """Configuration for frame-core data management."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to config.toml. If None, uses default locations.
        """
        self._config = self._load_config(config_path)
        
    def _load_config(self, config_path: Optional[Path]) -> dict:
        """Load configuration from file or use defaults."""
        defaults = {
            "frame": {
                "data_root": "./frame_data",
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
        
        # Try to find config file
        if config_path is None:
            # Try workspace config first
            workspace_config = Path.cwd() / "config" / "config.toml"
            if workspace_config.exists():
                config_path = workspace_config
            # Then try home directory
            elif (Path.home() / ".frame" / "config.toml").exists():
                config_path = Path.home() / ".frame" / "config.toml"
        
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

