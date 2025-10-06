# Frame-Core Test Suite

Comprehensive test suite for the frame-core package.

## Test Coverage

**Overall**: 85 tests, 3 skipped (CUDA tests on non-GPU systems)

### Coverage by Module

- **VoxelGrid** (`voxel_grid.py`): 94% coverage
  - Core data model and operations
  - Device management (CPU/GPU)
  - Channel access
  - Cloning and metadata

- **Storage** (`storage.py`): 92% coverage
  - VoxelLibrary reading and writing
  - Lazy loading
  - Parameter filtering
  - Round-trip data integrity

- **Dataset** (`dataset.py`): 98% coverage
  - PyTorch Dataset integration
  - DataLoader compatibility
  - Caching
  - Batch collation

- **Visualization** (13-24% coverage)
  - Visualization modules require display/rendering
  - Harder to test in automated CI
  - Manual testing recommended

## Test Files

### `test_voxel_grid.py` (19 tests)
Tests for the core `VoxelGrid` data model:
- Creation and validation
- Properties (shape, size, dtype)
- Channel access
- Device management (CPU/GPU)
- Cloning operations
- String representation

### `test_storage.py` (23 tests)
Tests for library storage backend:
- Creating libraries with `VoxelLibraryWriter`
- Reading libraries with `VoxelLibrary`
- Lazy loading behavior
- Parameter filtering
- Batch loading
- Round-trip data integrity
- Error handling

### `test_dataset.py` (19 tests)
Tests for PyTorch integration:
- `VoxelDataset` creation and access
- Index subsetting
- Transform functions
- Device placement
- Caching with `CachedVoxelDataset`
- Collation functions
- DataLoader integration
- Edge cases

### `test_utils.py` (24 tests)
Tests for edge cases and utilities:
- Different data types (float32, float64, numpy)
- Various grid sizes (8³ to 128³)
- Non-cubic grids
- Different voxel sizes
- Many channels
- Memory efficiency
- Parameter DataFrame operations
- Edge cases

### `conftest.py`
Shared pytest fixtures:
- `sample_voxel_grid`: 5-channel 32³ grid for testing
- `small_voxel_grid`: 3-channel 16³ grid for quick tests
- `temp_library_dir`: Temporary directory for library tests
- `sample_library`: Pre-populated library with 10 structures
- `device`: CPU or CUDA device if available

## Running Tests

### Run all tests:
```bash
cd /path/to/FRAME
uv run pytest packages/frame-core/tests/
```

### Run with verbose output:
```bash
uv run pytest packages/frame-core/tests/ -v
```

### Run with coverage:
```bash
uv run pytest packages/frame-core/tests/ --cov=frame_core --cov-report=term-missing
```

### Run specific test file:
```bash
uv run pytest packages/frame-core/tests/test_voxel_grid.py
```

### Run specific test:
```bash
uv run pytest packages/frame-core/tests/test_voxel_grid.py::TestVoxelGridCreation::test_create_basic_voxel_grid
```

### Run only fast tests (skip CUDA tests):
```bash
uv run pytest packages/frame-core/tests/ -m "not gpu"
```

## Test Categories

### Core Functionality (100% passing)
- ✅ VoxelGrid creation, properties, and operations
- ✅ Library creation and reading
- ✅ Lazy loading and parameter filtering
- ✅ PyTorch Dataset and DataLoader integration
- ✅ Data type handling and conversions

### Performance and Memory
- ✅ Memory efficiency (no unnecessary copies)
- ✅ Lazy loading (only load what's needed)
- ✅ Caching behavior
- ✅ Batch operations

### Edge Cases
- ✅ Empty indices
- ✅ Single item datasets
- ✅ Very small grids (2³)
- ✅ Large grids (128³)
- ✅ Non-cubic grids
- ✅ Many channels (20+)
- ✅ Various voxel sizes

### Error Handling
- ✅ Invalid data shapes
- ✅ Invalid voxel sizes
- ✅ Nonexistent libraries
- ✅ Missing files
- ✅ Index out of range

### GPU Support
- ⏭️ CUDA tests skipped on non-GPU systems
- ✅ All GPU tests pass on CUDA-enabled systems

## Future Test Additions

### Visualization Tests
The visualization modules (`visualize_napari.py`, `visualize_pyvista.py`, `visualize_batch.py`) have lower coverage because they require:
- Display/rendering capabilities
- Interactive GUI testing
- Image comparison

**Recommended**: Manual testing for visualization features

### Integration Tests
Consider adding:
- Large-scale library tests (1000+ structures)
- Performance benchmarks
- Memory profiling tests
- Multi-threaded DataLoader tests
- Compression ratio measurements

### Property-Based Testing
Consider using `hypothesis` for:
- Random grid sizes
- Random channel counts
- Random parameter queries
- Fuzz testing data types

## Notes

- Tests use temporary directories (cleaned up automatically)
- CUDA tests are skipped on CPU-only systems
- Warnings from Python's copy module are expected (deprecation in itertools)
- All tests are independent and can run in any order

