"""Lineage tracking for libraries."""

import json
from pathlib import Path
from typing import Optional

from .library import LibraryManager, Library


class LineageTracker:
    """Track lineage of derived libraries."""
    
    def __init__(self):
        """Initialize lineage tracker."""
        self.manager = LibraryManager()
    
    def get_lineage(self, library_uuid: str) -> list[Library]:
        """Get full lineage chain for a library.
        
        Args:
            library_uuid: UUID of library to trace
        
        Returns:
            List of Library objects from ancestor to current
        """
        lineage = []
        current_uuid = library_uuid
        
        while current_uuid:
            library = self.manager.get_library(current_uuid)
            if not library:
                break
            
            lineage.insert(0, library)
            current_uuid = library.derived_from
        
        return lineage
    
    def get_descendants(self, library_uuid: str) -> list[Library]:
        """Get all descendants of a library.
        
        Args:
            library_uuid: UUID of library
        
        Returns:
            List of descendant Library objects
        """
        descendants = []
        all_libraries = self.manager.list_libraries()
        
        for library in all_libraries:
            if library.derived_from == library_uuid:
                descendants.append(library)
                # Recursively get descendants
                descendants.extend(self.get_descendants(library.uuid))
        
        return descendants
    
    def get_lineage_info(self, library_uuid: str) -> Optional[dict]:
        """Get detailed lineage information for a library.
        
        Args:
            library_uuid: UUID of library
        
        Returns:
            Lineage information dictionary or None if no lineage file
        """
        library = self.manager.get_library(library_uuid)
        if not library:
            return None
        
        lineage_path = library.path / "lineage.json"
        if not lineage_path.exists():
            return None
        
        with open(lineage_path, "r") as f:
            return json.load(f)

