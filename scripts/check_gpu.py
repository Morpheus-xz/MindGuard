"""Check GPU/MPS availability for MindGuard."""

import sys

sys.path.insert(0, ".")

import torch


def check_device():
    """Check and display available compute devices."""
    print("=" * 50)
    print("MindGuard Device Check")
    print("=" * 50)
    print(f"PyTorch version: {torch.__version__}")
    print()

    # Check MPS (Apple Silicon)
    print("Checking Apple MPS (Metal)...")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("  ✅ MPS is AVAILABLE")
        if torch.backends.mps.is_built():
            print("  ✅ MPS is BUILT into PyTorch")
        device = torch.device("mps")

        # Quick test
        try:
            x = torch.ones(3, 3, device=device)
            y = x * 2
            print("  ✅ MPS tensor operations work")
        except Exception as e:
            print(f"  ⚠️ MPS test failed: {e}")
    else:
        print("  ❌ MPS not available")

    print()

    # Check CUDA
    print("Checking NVIDIA CUDA...")
    if torch.cuda.is_available():
        print("  ✅ CUDA is AVAILABLE")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  ❌ CUDA not available")

    print()

    # Final recommendation
    print("=" * 50)
    print("RECOMMENDATION")
    print("=" * 50)

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("🍎 Use MPS (Apple Silicon) for training")
        print("   Your M4 MacBook will use the Metal GPU")
        selected = "mps"
    elif torch.cuda.is_available():
        print("🟢 Use CUDA for training")
        selected = "cuda"
    else:
        print("⚠️ No GPU detected - using CPU")
        print("   Training will be slower")
        selected = "cpu"

    print()
    print(f"Selected device: {selected}")
    print("=" * 50)

    return selected


if __name__ == "__main__":
    check_device()