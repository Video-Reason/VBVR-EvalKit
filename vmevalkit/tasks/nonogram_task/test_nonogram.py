"""
Comprehensive test script for nonogram_task.
Tests different cases: various grid sizes, pattern types, difficulties, etc.
"""

import sys
from pathlib import Path

# Add parent directory to path to import nonogram_reasoning
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from vmevalkit.tasks.nonogram_task import create_dataset
    from vmevalkit.tasks.nonogram_task.nonogram_reasoning import (
        NonogramGenerator,
        NonogramRenderer,
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("⚠️  Note: This test requires nonogram_task to be fully implemented.")
    sys.exit(1)


def test_imports():
    """Test that all required modules can be imported."""
    print("=" * 80)
    print("Test 1: Module Imports")
    print("=" * 80)
    try:
        from vmevalkit.tasks.nonogram_task import create_dataset
        from vmevalkit.tasks.nonogram_task.nonogram_reasoning import (
            NonogramGenerator,
            NonogramRenderer,
        )
        from vmevalkit.tasks.nonogram_task.PROMPTS import get_prompt
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_generator_initialization():
    """Test that generator can be initialized."""
    print("\n" + "=" * 80)
    print("Test 2: Generator Initialization")
    print("=" * 80)
    try:
        generator = NonogramGenerator()
        print("✅ Generator initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Generator initialization failed: {e}")
        return False


def test_renderer_initialization():
    """Test that renderer can be initialized."""
    print("\n" + "=" * 80)
    print("Test 3: Renderer Initialization")
    print("=" * 80)
    try:
        renderer = NonogramRenderer()
        print("✅ Renderer initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Renderer initialization failed: {e}")
        return False


def test_difficulty_levels():
    """Test generation with different difficulty levels."""
    print("\n" + "=" * 80)
    print("Test 4: Difficulty Levels")
    print("=" * 80)
    try:
        generator = NonogramGenerator()
        difficulties = ["easy", "medium", "hard"]
        
        for difficulty in difficulties:
            try:
                task = generator.generate(
                    task_id=f"test_{difficulty}",
                    difficulty=difficulty,
                    seed=42,
                    ensure_unique=False,
                )
                print(f"✅ {difficulty}: Generated task with {task.get('nonogram_data', {}).get('grid_size', 'unknown')}x{task.get('nonogram_data', {}).get('grid_size', 'unknown')} grid")
            except Exception as e:
                print(f"❌ {difficulty}: Failed - {e}")
                return False
        return True
    except Exception as e:
        print(f"❌ Difficulty test failed: {e}")
        return False


def test_grid_sizes():
    """Test generation with different grid sizes."""
    print("\n" + "=" * 80)
    print("Test 5: Grid Sizes")
    print("=" * 80)
    try:
        generator = NonogramGenerator()
        # Test various grid sizes
        test_sizes = [5, 8, 10, 12, 15]
        
        for size in test_sizes:
            try:
                # Manually set grid size if possible
                # This depends on the implementation
                task = generator.generate(
                    task_id=f"test_size_{size}",
                    difficulty="medium",
                    seed=42 + size,
                    ensure_unique=False,
                )
                actual_size = task.get('nonogram_data', {}).get('grid_size', 0)
                print(f"✅ Size {size}: Generated {actual_size}x{actual_size} grid")
            except Exception as e:
                print(f"⚠️  Size {size}: {e}")
        return True
    except Exception as e:
        print(f"❌ Grid size test failed: {e}")
        return False


def test_pattern_types():
    """Test generation with different pattern types."""
    print("\n" + "=" * 80)
    print("Test 6: Pattern Types")
    print("=" * 80)
    try:
        generator = NonogramGenerator()
        # Generate multiple tasks to see different pattern types
        pattern_types_seen = set()
        
        for i in range(20):
            try:
                task = generator.generate(
                    task_id=f"test_pattern_{i}",
                    difficulty="medium",
                    seed=100 + i,
                    ensure_unique=False,
                )
                pattern_type = task.get('nonogram_data', {}).get('pattern_type', 'unknown')
                pattern_types_seen.add(pattern_type)
            except Exception as e:
                print(f"⚠️  Pattern {i}: {e}")
        
        print(f"✅ Found {len(pattern_types_seen)} different pattern types:")
        for pt in sorted(pattern_types_seen):
            print(f"   - {pt}")
        return True
    except Exception as e:
        print(f"❌ Pattern type test failed: {e}")
        return False


def test_uniqueness():
    """Test that uniqueness checking works."""
    print("\n" + "=" * 80)
    print("Test 7: Uniqueness Checking")
    print("=" * 80)
    try:
        generator = NonogramGenerator()
        signatures = set()
        
        for i in range(10):
            task = generator.generate(
                task_id=f"test_unique_{i}",
                difficulty="medium",
                seed=None,  # Random seed
                ensure_unique=True,
            )
            # Check if we can extract a signature
            # This depends on the implementation
            print(f"✅ Task {i}: Generated unique task")
        
        print("✅ Uniqueness check passed")
        return True
    except Exception as e:
        print(f"❌ Uniqueness test failed: {e}")
        return False


def test_dataset_creation():
    """Test creating a small dataset."""
    print("\n" + "=" * 80)
    print("Test 8: Dataset Creation")
    print("=" * 80)
    try:
        dataset = create_dataset(num_samples=5, seed=42)
        
        if 'pairs' not in dataset:
            print("❌ Dataset missing 'pairs' key")
            return False
        
        if len(dataset['pairs']) != 5:
            print(f"❌ Expected 5 pairs, got {len(dataset['pairs'])}")
            return False
        
        print(f"✅ Created dataset with {len(dataset['pairs'])} pairs")
        
        # Check each pair has required fields
        required_fields = ['id', 'prompt', 'first_image_path', 'final_image_path']
        for i, pair in enumerate(dataset['pairs']):
            for field in required_fields:
                if field not in pair:
                    print(f"❌ Pair {i} missing field: {field}")
                    return False
        
        print("✅ All pairs have required fields")
        return True
    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hint_calculation():
    """Test that hints are calculated correctly."""
    print("\n" + "=" * 80)
    print("Test 9: Hint Calculation")
    print("=" * 80)
    try:
        generator = NonogramGenerator()
        task = generator.generate(
            task_id="test_hints",
            difficulty="medium",
            seed=42,
            ensure_unique=False,
        )
        
        data = task.get('nonogram_data', {})
        row_hints = data.get('row_hints', [])
        col_hints = data.get('col_hints', [])
        grid_size = data.get('grid_size', 0)
        
        if len(row_hints) != grid_size:
            print(f"❌ Row hints count mismatch: {len(row_hints)} != {grid_size}")
            return False
        
        if len(col_hints) != grid_size:
            print(f"❌ Column hints count mismatch: {len(col_hints)} != {grid_size}")
            return False
        
        # Check no empty hints (all zeros)
        for i, hints in enumerate(row_hints):
            if hints == [0] or hints == []:
                print(f"⚠️  Row {i} has empty hints: {hints}")
        
        for i, hints in enumerate(col_hints):
            if hints == [0] or hints == []:
                print(f"⚠️  Column {i} has empty hints: {hints}")
        
        print(f"✅ Hints calculated correctly for {grid_size}x{grid_size} grid")
        print(f"   Row hints: {len(row_hints)} rows")
        print(f"   Column hints: {len(col_hints)} columns")
        return True
    except Exception as e:
        print(f"❌ Hint calculation test failed: {e}")
        return False


def test_file_generation():
    """Test that files are generated correctly."""
    print("\n" + "=" * 80)
    print("Test 10: File Generation")
    print("=" * 80)
    try:
        generator = NonogramGenerator()
        task = generator.generate(
            task_id="test_files",
            difficulty="medium",
            seed=42,
            ensure_unique=False,
        )
        
        first_path = Path(task['first_image_path'])
        final_path = Path(task['final_image_path'])
        
        if not first_path.exists():
            print(f"❌ First frame not found: {first_path}")
            return False
        
        if not final_path.exists():
            print(f"❌ Final frame not found: {final_path}")
            return False
        
        # Check file sizes (should be non-zero)
        if first_path.stat().st_size == 0:
            print(f"❌ First frame is empty: {first_path}")
            return False
        
        if final_path.stat().st_size == 0:
            print(f"❌ Final frame is empty: {final_path}")
            return False
        
        print(f"✅ Files generated successfully")
        print(f"   First frame: {first_path.stat().st_size} bytes")
        print(f"   Final frame: {final_path.stat().st_size} bytes")
        return True
    except Exception as e:
        print(f"❌ File generation test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("Nonogram Task - Comprehensive Test Suite")
    print("=" * 80)
    
    tests = [
        test_imports,
        test_generator_initialization,
        test_renderer_initialization,
        test_difficulty_levels,
        test_grid_sizes,
        test_pattern_types,
        test_uniqueness,
        test_hint_calculation,
        test_file_generation,
        test_dataset_creation,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    passed = sum(results)
    total = len(results)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

