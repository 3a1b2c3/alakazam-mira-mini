#!/usr/bin/env python3
import inspect
from mira.world_model.config import WorldModelInferenceConfig

print("WorldModelInferenceConfig:")
print("=" * 70)

# Get signature
sig = inspect.signature(WorldModelInferenceConfig.__init__)
print("\nSignature:")
print(sig)

# Get defaults
print("\nDefault values:")
cfg = WorldModelInferenceConfig()
for k, v in cfg.__dict__.items():
    print(f"  {k}: {v}")

# Show field types if available
print("\nAll fields:")
if hasattr(WorldModelInferenceConfig, '__annotations__'):
    for k, v in WorldModelInferenceConfig.__annotations__.items():
        print(f"  {k}: {v}")
