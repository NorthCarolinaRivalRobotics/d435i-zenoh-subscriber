#!/usr/bin/env python3
"""
Simple profiling utilities for performance analysis.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, List
from collections import defaultdict
import statistics


class ProfilerResults:
    """Stores and analyzes profiling results."""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.counts: Dict[str, int] = defaultdict(int)
    
    def add_timing(self, name: str, duration: float) -> None:
        """Add a timing measurement."""
        self.timings[name].append(duration)
        self.counts[name] += 1
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a specific timer."""
        if name not in self.timings or not self.timings[name]:
            return {}
        
        times = self.timings[name]
        return {
            'count': len(times),
            'total': sum(times),
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'min': min(times),
            'max': max(times),
            'std': statistics.stdev(times) if len(times) > 1 else 0.0
        }
    
    def print_summary(self, min_calls: int = 5) -> None:
        """Print a summary of all timings."""
        print("\n" + "="*60)
        print("PROFILING SUMMARY")
        print("="*60)
        
        # Sort by total time
        items = [(name, self.get_stats(name)) for name in self.timings.keys()]
        items = [(name, stats) for name, stats in items if stats.get('count', 0) >= min_calls]
        items.sort(key=lambda x: x[1].get('total', 0), reverse=True)
        
        print(f"{'Component':<25} {'Calls':<8} {'Total(s)':<10} {'Mean(ms)':<10} {'Max(ms)':<10} {'FPS':<8}")
        print("-" * 60)
        
        for name, stats in items:
            if not stats:
                continue
            fps = 1.0 / stats['mean'] if stats['mean'] > 0 else 0
            print(f"{name:<25} {stats['count']:<8} {stats['total']:<10.3f} "
                  f"{stats['mean']*1000:<10.1f} {stats['max']*1000:<10.1f} {fps:<8.1f}")
        
        print("-" * 60)
        
        # Show percentage breakdown
        total_time = sum(stats.get('total', 0) for _, stats in items)
        if total_time > 0:
            print("\nTime breakdown:")
            for name, stats in items:
                if stats:
                    percentage = (stats['total'] / total_time) * 100
                    print(f"  {name:<25} {percentage:>6.1f}%")


class SimpleProfiler:
    """Simple profiler for measuring execution times."""
    
    def __init__(self):
        self.results = ProfilerResults()
        self.enabled = True
    
    def enable(self):
        """Enable profiling."""
        self.enabled = True
    
    def disable(self):
        """Disable profiling."""
        self.enabled = False
    
    @contextmanager
    def timer(self, name: str):
        """Context manager for timing code blocks."""
        if not self.enabled:
            yield
            return
            
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            self.results.add_timing(name, duration)
    
    def print_results(self, min_calls: int = 5):
        """Print profiling results."""
        self.results.print_summary(min_calls)
    
    def reset(self):
        """Reset all profiling data."""
        self.results = ProfilerResults()


# Global profiler instance
profiler = SimpleProfiler() 