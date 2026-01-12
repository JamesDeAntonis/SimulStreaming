#!/usr/bin/env python3
"""
Benchmark script to compare inference speed with and without torch.compile()

This script tests the speed difference between compiled and non-compiled models
by running multiple inference iterations and timing them.
"""

import argparse
import time
import logging
import sys
import torch
import numpy as np

# Setup logging
logging.basicConfig(level=logging.WARNING)  # Suppress most logs for cleaner output
logger = logging.getLogger(__name__)

from simul_whisper.config import AlignAttConfig
from simul_whisper.simul_whisper import PaddedAlignAttWhisper

def generate_test_audio(duration_seconds=5.0, sample_rate=16000):
    """Generate a simple test audio signal (sine wave)"""
    num_samples = int(duration_seconds * sample_rate)
    # Generate a simple sine wave at 440 Hz (A note)
    t = np.linspace(0, duration_seconds, num_samples)
    audio = np.sin(2 * np.pi * 440 * t) * 0.3
    # Add some variation to make it more realistic
    audio += np.random.normal(0, 0.01, num_samples)
    return torch.from_numpy(audio.astype(np.float32)).float()

def benchmark_inference(cfg, num_iterations=10, warmup_iterations=2):
    """Benchmark inference speed with given config"""
    logger.info(f"Initializing model with use_compile={cfg.use_compile}")
    
    # Create model instance
    model = PaddedAlignAttWhisper(cfg)
    
    # Generate test audio
    test_audio = generate_test_audio(duration_seconds=3.0)
    
    # Warmup iterations
    logger.info(f"Running {warmup_iterations} warmup iterations...")
    for _ in range(warmup_iterations):
        model.refresh_segment(complete=True)
        model.insert_audio(test_audio)
        try:
            model.infer(is_last=False)
        except Exception as e:
            logger.warning(f"Warmup iteration failed: {e}")
        model.refresh_segment(complete=True)
    
    # Actual timing
    logger.info(f"Running {num_iterations} timed iterations...")
    times = []
    
    for i in range(num_iterations):
        model.refresh_segment(complete=True)
        model.insert_audio(test_audio)
        
        # Time just the inference step
        start_time = time.perf_counter()
        try:
            model.infer(is_last=False)
        except Exception as e:
            logger.warning(f"Iteration {i} failed: {e}")
            continue
        end_time = time.perf_counter()
        
        elapsed = end_time - start_time
        times.append(elapsed)
    
    if not times:
        return None, None
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    return avg_time, {
        'times': times,
        'std': std_time,
        'min': min_time,
        'max': max_time
    }

def main():
    parser = argparse.ArgumentParser(description="Benchmark torch.compile() speedup")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./large-v3.pt",
        help="Path to Whisper model file",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations to run for timing (default: 10)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup iterations (default: 2)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code (default: en)",
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("Benchmarking torch.compile() Speedup")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Iterations: {args.iterations} (with {args.warmup} warmup)")
    print(f"Language: {args.language}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"torch.compile available: {hasattr(torch, 'compile')}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print("=" * 70)
    print()
    
    # Common config
    base_config = {
        "model_path": args.model_path,
        "language": args.language,
        "segment_length": 1.0,
        "frame_threshold": 25,
        "audio_max_len": 30.0,
        "audio_min_len": 0.0,
        "cif_ckpt_path": None,
        "decoder_type": "greedy",
        "beam_size": 1,
        "task": "transcribe",
        "never_fire": False,
        "init_prompt": None,
        "static_init_prompt": None,
        "max_context_tokens": None,
        "logdir": None,
        "use_half_precision": None,  # Auto-detect
        "enable_cudnn_benchmark": True,
    }
    
    # Benchmark WITHOUT compilation
    print("1. Benchmarking WITHOUT torch.compile()...")
    print("-" * 70)
    cfg_no_compile = AlignAttConfig(**base_config, use_compile=False)
    avg_no_compile, stats_no_compile = benchmark_inference(
        cfg_no_compile, 
        num_iterations=args.iterations,
        warmup_iterations=args.warmup
    )
    
    if avg_no_compile is None:
        print("ERROR: No successful iterations without compilation")
        return 1
    
    print(f"Average time: {avg_no_compile*1000:.2f} ms")
    print(f"Std deviation: {stats_no_compile['std']*1000:.2f} ms")
    print(f"Min: {stats_no_compile['min']*1000:.2f} ms, Max: {stats_no_compile['max']*1000:.2f} ms")
    print()
    
    # Benchmark WITH compilation
    print("2. Benchmarking WITH torch.compile()...")
    print("-" * 70)
    if not hasattr(torch, 'compile'):
        print("WARNING: torch.compile() not available (requires PyTorch 2.0+)")
        print("Skipping compiled benchmark")
        print()
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Without compilation: {avg_no_compile*1000:.2f} ms")
        print("Compilation: Not available")
        return 0
    
    cfg_with_compile = AlignAttConfig(**base_config, use_compile=True)
    avg_with_compile, stats_with_compile = benchmark_inference(
        cfg_with_compile,
        num_iterations=args.iterations,
        warmup_iterations=args.warmup
    )
    
    if avg_with_compile is None:
        print("ERROR: No successful iterations with compilation")
        return 1
    
    print(f"Average time: {avg_with_compile*1000:.2f} ms")
    print(f"Std deviation: {stats_with_compile['std']*1000:.2f} ms")
    print(f"Min: {stats_with_compile['min']*1000:.2f} ms, Max: {stats_with_compile['max']*1000:.2f} ms")
    print()
    
    # Results summary
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Without compilation: {avg_no_compile*1000:.2f} ms")
    print(f"With compilation:    {avg_with_compile*1000:.2f} ms")
    print()
    
    speedup = avg_no_compile / avg_with_compile
    improvement_pct = (1 - avg_with_compile / avg_no_compile) * 100
    
    print(f"Speedup: {speedup:.2f}x")
    print(f"Improvement: {improvement_pct:.1f}% faster")
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
