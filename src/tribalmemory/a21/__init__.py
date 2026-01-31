"""A2.1 Interface-First Implementation.

This module provides a highly abstracted, plugin-based architecture for
tribal memory. Key design principles:

1. **Provider Pattern**: All backends (embeddings, storage, timestamps) are
   swappable providers that implement standard interfaces.

2. **Dependency Injection**: Components receive their dependencies through
   a central container, making testing and configuration easier.

3. **Configuration-Driven**: Setup is driven by configuration objects,
   not hardcoded values.

4. **Forward Compatible**: Interfaces are designed to accommodate future
   features (multi-tenancy, sharding, replication) without breaking changes.

Usage:
    from tribalmemory.a21 import MemorySystem, SystemConfig
    
    config = SystemConfig.from_env()
    system = MemorySystem(config)
    
    await system.remember("Joe prefers TypeScript")
    results = await system.recall("What language?")
"""

from .system import MemorySystem
from .config import SystemConfig, EmbeddingConfig, StorageConfig
from .container import Container

__all__ = [
    "MemorySystem",
    "SystemConfig",
    "EmbeddingConfig",
    "StorageConfig",
    "Container",
]
