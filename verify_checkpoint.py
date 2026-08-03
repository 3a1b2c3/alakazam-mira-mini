import torch
import sys

checkpoint_path = r"C:\workspace\world\alakazam-mira-mini\outputs\finetune_ch\checkpoint-49000\checkpoint.pth"

try:
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    print(f"\n[OK] Checkpoint loaded successfully")
    print(f"Checkpoint type: {type(checkpoint)}")

    if isinstance(checkpoint, dict):
        print(f"\nKeys in checkpoint: {list(checkpoint.keys())}")
        for key in checkpoint.keys():
            if isinstance(checkpoint[key], torch.Tensor):
                print(f"  {key}: shape={checkpoint[key].shape}, dtype={checkpoint[key].dtype}, device={checkpoint[key].device}")
            elif isinstance(checkpoint[key], dict):
                print(f"  {key}: dict with {len(checkpoint[key])} items")
            else:
                print(f"  {key}: {type(checkpoint[key])}")

    print("\n[OK] Checkpoint is COMPLETE and valid")
    sys.exit(0)

except Exception as e:
    print(f"\n[ERROR] Loading checkpoint failed: {e}")
    sys.exit(1)
