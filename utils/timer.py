"""
Timer utilities for QR code robust reading project.
Provides timing functions and FPS calculation.
"""

import time
from typing import Optional


class Timer:
    """Simple timer for measuring execution time."""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.perf_counter()
        self.end_time = None
    
    def stop(self) -> float:
        """
        Stop the timer and return elapsed time.
        
        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        
        self.end_time = time.perf_counter()
        return self.elapsed()
    
    def elapsed(self) -> float:
        """
        Get elapsed time without stopping the timer.
        
        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        
        end_time = self.end_time if self.end_time is not None else time.perf_counter()
        return end_time - self.start_time
    
    def elapsed_ms(self) -> float:
        """
        Get elapsed time in milliseconds.
        
        Returns:
            Elapsed time in milliseconds
        """
        return self.elapsed() * 1000.0


def now_ms() -> float:
    """
    Get current time in milliseconds.
    
    Returns:
        Current time in milliseconds since epoch
    """
    return time.perf_counter() * 1000.0


def elapsed_ms(start_time_ms: float) -> float:
    """
    Calculate elapsed time from start time.
    
    Args:
        start_time_ms: Start time in milliseconds
        
    Returns:
        Elapsed time in milliseconds
    """
    return now_ms() - start_time_ms


def calculate_fps(total_time_seconds: float, num_items: int) -> float:
    """
    Calculate FPS (frames per second) or items per second.
    
    Args:
        total_time_seconds: Total processing time in seconds
        num_items: Number of items processed
        
    Returns:
        FPS (items per second)
    """
    if total_time_seconds <= 0:
        return 0.0
    return num_items / total_time_seconds


class FPSCounter:
    """FPS counter for batch processing."""
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> None:
        """Reset the counter."""
        self.total_time = 0.0
        self.total_items = 0
    
    def add_batch(self, batch_time_seconds: float, batch_size: int) -> None:
        """
        Add a batch processing time.
        
        Args:
            batch_time_seconds: Time taken for this batch
            batch_size: Number of items in this batch
        """
        self.total_time += batch_time_seconds
        self.total_items += batch_size
    
    def get_fps(self) -> float:
        """
        Get current FPS.
        
        Returns:
            Current FPS (items per second)
        """
        return calculate_fps(self.total_time, self.total_items)
    
    def get_avg_time_ms(self) -> float:
        """
        Get average processing time per item in milliseconds.
        
        Returns:
            Average time per item in milliseconds
        """
        if self.total_items == 0:
            return 0.0
        return (self.total_time * 1000.0) / self.total_items
