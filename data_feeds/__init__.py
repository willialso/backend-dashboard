# data_feeds/__init__.py

"""
Data Feeds Package for Atticus Platform

This package handles connections to external exchanges to fetch real-time
market data, primarily BTC price feeds.
"""

__version__ = "1.0.0"
__author__ = "Atticus Team"

# Import the main classes from this package
from .data_feed_manager import DataFeedManager, PriceData

# Define the public API
__all__ = [
    "DataFeedManager",
    "PriceData",  # Also export PriceData as it might be useful for type hints
]
