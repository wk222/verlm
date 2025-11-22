#!/usr/bin/env python
"""
Test ADPO Installation

This script tests if ADPO is correctly integrated into VERL.
Run this before starting ADPO training to verify your setup.
"""

import sys
import importlib


def test_imports():
    """Test if all ADPO modules can be imported."""
    print("Testing ADPO module imports...")
    
    tests = [
        ("verl.trainer.adpo", "ADPO main module"),
        ("verl.trainer.adpo.core_algos", "ADPO core algorithms"),
        ("verl.trainer.adpo.ray_trainer", "ADPO Ray trainer"),
        ("verl.trainer.adpo.reward", "ADPO reward functions"),
        ("verl.trainer.adpo.utils", "ADPO utilities"),
        ("verl.trainer.main_adpo", "ADPO main entry point"),
    ]
    
    passed = 0
    failed = 0
    
    for module_name, description in tests:
        try:
            importlib.import_module(module_name)
            print(f"  ✅ {description} ({module_name})")
            passed += 1
        except ImportError as e:
            print(f"  ❌ {description} ({module_name})")
            print(f"     Error: {e}")
            failed += 1
    
    print(f"\nImport test results: {passed} passed, {failed} failed")
    return failed == 0


def test_registry():
    """Test if ADPO is properly registered."""
    print("\nTesting ADPO registry...")
    
    try:
        from verl.trainer.ppo.core_algos import (
            ADV_ESTIMATOR_REGISTRY,
            POLICY_LOSS_REGISTRY,
        )
        
        # Check advantage estimator
        if "adpo" in ADV_ESTIMATOR_REGISTRY:
            print("  ✅ ADPO advantage estimator registered")
        else:
            print("  ❌ ADPO advantage estimator NOT registered")
            print(f"     Available: {list(ADV_ESTIMATOR_REGISTRY.keys())}")
            return False
        
        # Check policy loss
        if "adpo" in POLICY_LOSS_REGISTRY:
            print("  ✅ ADPO policy loss registered")
        else:
            print("  ❌ ADPO policy loss NOT registered")
            print(f"     Available: {list(POLICY_LOSS_REGISTRY.keys())}")
            return False
        
        return True
    
    except Exception as e:
        print(f"  ❌ Registry test failed: {e}")
        return False


def test_config():
    """Test if ADPO config file exists."""
    print("\nTesting ADPO configuration...")
    
    import os
    
    config_path = "verl/trainer/config/adpo_trainer.yaml"
    if os.path.exists(config_path):
        print(f"  ✅ ADPO config file found: {config_path}")
        return True
    else:
        print(f"  ❌ ADPO config file NOT found: {config_path}")
        return False


def test_examples():
    """Test if example scripts exist."""
    print("\nTesting ADPO examples...")
    
    import os
    
    examples = [
        "examples/run_adpo_gsm8k.sh",
        "examples/run_adpo_fixed_anchor.sh",
        "examples/run_adpo_ema.sh",
        "examples/quickstart_adpo.sh",
        "examples/adpo_example_config.py",
    ]
    
    all_exist = True
    for example in examples:
        if os.path.exists(example):
            print(f"  ✅ {example}")
        else:
            print(f"  ❌ {example} NOT found")
            all_exist = False
    
    return all_exist


def test_optional_dependencies():
    """Test optional dependencies for good_accuracy reward."""
    print("\nTesting optional dependencies (for good_accuracy reward)...")
    
    optional_deps = [
        ("latex2sympy2_extended", "LaTeX to SymPy conversion"),
        ("math_verify", "Math answer verification"),
    ]
    
    all_available = True
    for dep, description in optional_deps:
        try:
            importlib.import_module(dep)
            print(f"  ✅ {description} ({dep})")
        except ImportError:
            print(f"  ⚠️  {description} ({dep}) NOT installed (optional)")
            all_available = False
    
    if not all_available:
        print("\n  Note: To use good_accuracy reward, install with:")
        print("    pip install latex2sympy2_extended math_verify")
    
    return True  # These are optional, so always return True


def test_trainer_creation():
    """Test if RayADPOTrainer can be instantiated."""
    print("\nTesting ADPO trainer creation...")
    
    try:
        from verl.trainer.adpo.ray_trainer import RayADPOTrainer
        print("  ✅ RayADPOTrainer class can be imported")
        
        # Check if it has required methods
        required_methods = ["__init__", "fit", "init_workers"]
        for method in required_methods:
            if hasattr(RayADPOTrainer, method):
                print(f"  ✅ RayADPOTrainer.{method} exists")
            else:
                print(f"  ❌ RayADPOTrainer.{method} NOT found")
                return False
        
        return True
    
    except Exception as e:
        print(f"  ❌ Trainer creation test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("ADPO Installation Test")
    print("=" * 50)
    print()
    
    results = {
        "Imports": test_imports(),
        "Registry": test_registry(),
        "Config": test_config(),
        "Examples": test_examples(),
        "Optional Dependencies": test_optional_dependencies(),
        "Trainer Creation": test_trainer_creation(),
    }
    
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All tests passed! ADPO is ready to use.")
        print("\nNext steps:")
        print("  1. Read the documentation: verl/trainer/adpo/README.md")
        print("  2. Run quickstart: bash examples/quickstart_adpo.sh")
        print("  3. Or try an example: bash examples/run_adpo_gsm8k.sh")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("  1. Make sure you're in the verlm/ directory")
        print("  2. Check if all ADPO files are present")
        print("  3. Verify PYTHONPATH is set correctly")
    
    print("=" * 50)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

