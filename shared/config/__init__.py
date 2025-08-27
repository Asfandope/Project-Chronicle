"""
Configuration management for layout understanding system.
"""

from .brand_loader import BrandConfigLoader, get_brand_loader, load_brand_config

__all__ = ["BrandConfigLoader", "get_brand_loader", "load_brand_config"]
