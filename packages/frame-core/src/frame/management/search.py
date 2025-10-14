"""Search and query functionality for libraries and experiments."""

from typing import Any, Optional, Callable

from .library import LibraryManager, Library
from .experiment import ExperimentManager, Experiment


class LibrarySearch:
    """Search and query libraries."""
    
    def __init__(self):
        """Initialize library search."""
        self.manager = LibraryManager()
    
    def query(
        self,
        structure_type: Optional[str] = None,
        params: Optional[dict[str, dict[str, Any]]] = None,
        n_structures: Optional[dict[str, int]] = None,
        tags: Optional[list[str]] = None,
    ) -> list[Library]:
        """Query libraries with filters.
        
        Args:
            structure_type: Filter by structure type
            params: Parameter filters (e.g., {'shell1_radius_nm': {'$gt': 50}})
            n_structures: Number of structures filter (e.g., {'$gt': 1000})
            tags: Filter by tags
        
        Returns:
            List of matching Library objects
        """
        # Start with all libraries matching tags
        libraries = self.manager.list_libraries(tags=tags)
        
        # Filter by structure type
        if structure_type:
            libraries = [lib for lib in libraries if lib.structure_type == structure_type]
        
        # Filter by number of structures
        if n_structures:
            libraries = self._apply_comparison_filter(
                libraries,
                lambda lib: lib.n_structures,
                n_structures
            )
        
        # TODO: Parameter filtering would require loading parameter data
        # For now, we skip parameter filtering at the library level
        
        return libraries
    
    def _apply_comparison_filter(
        self,
        items: list,
        getter: Callable,
        filter_dict: dict[str, Any]
    ) -> list:
        """Apply comparison filters.
        
        Args:
            items: Items to filter
            getter: Function to extract value from item
            filter_dict: Filter specification (e.g., {'$gt': 50, '$lt': 100})
        
        Returns:
            Filtered list
        """
        result = []
        for item in items:
            value = getter(item)
            passes = True
            
            for op, threshold in filter_dict.items():
                if op == "$gt" and not (value > threshold):
                    passes = False
                elif op == "$gte" and not (value >= threshold):
                    passes = False
                elif op == "$lt" and not (value < threshold):
                    passes = False
                elif op == "$lte" and not (value <= threshold):
                    passes = False
                elif op == "$eq" and not (value == threshold):
                    passes = False
                elif op == "$ne" and not (value != threshold):
                    passes = False
            
            if passes:
                result.append(item)
        
        return result


class ExperimentSearch:
    """Search and query experiments."""
    
    def __init__(self):
        """Initialize experiment search."""
        self.manager = ExperimentManager()
    
    def query(
        self,
        model_type: Optional[str] = None,
        library_uuid: Optional[str] = None,
        tags: Optional[list[str]] = None,
        status: Optional[str] = None,
    ) -> list[Experiment]:
        """Query experiments with filters.
        
        Args:
            model_type: Filter by model type
            library_uuid: Filter by library UUID
            tags: Filter by tags
            status: Filter by status
        
        Returns:
            List of matching Experiment objects
        """
        experiments = self.manager.list_experiments(
            model_type=model_type,
            tags=tags,
            status=status
        )
        
        # Filter by library UUID
        if library_uuid:
            experiments = [exp for exp in experiments if exp.library_uuid == library_uuid]
        
        return experiments

