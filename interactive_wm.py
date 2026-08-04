#!/usr/bin/env python3
"""Interactive world model player - explore checkpoint predictions"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import sys
import json
import tempfile
import tarfile
from pathlib import Path
from PIL import Image
import numpy as np

try:
    from torchcodec.decoders import VideoDecoder
except ImportError:
    print("ERROR: torchcodec not available")
    sys.exit(1)

class WorldModelPlayer:
    def __init__(self, ckpt_path, data_index):
        self.ckpt_path = Path(ckpt_path)
        self.data_index = Path(data_index)
        self.ckpt = None
        self.data_root = None
        self.entries = []
        self.current_frames = None
        self.current_entry = None

        self.load_checkpoint()
        self.load_data_index()

    def load_checkpoint(self):
        """Load model checkpoint"""
        if not self.ckpt_path.exists():
            print(f"ERROR: {self.ckpt_path} not found")
            sys.exit(1)

        print(f"Loading checkpoint from {self.ckpt_path.name}...")
        self.ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        print(f"✓ Loaded (iter={self.ckpt['iter_num']}, loss={self.ckpt['loss_diffusion']:.4f})")

    def load_data_index(self):
        """Load dataset index"""
        if not self.data_index.exists():
            print(f"ERROR: {self.data_index} not found")
            sys.exit(1)

        idx = json.load(open(self.data_index))
        self.data_root = self.data_index.parent
        self.entries = idx["entries"]
        print(f"✓ Loaded {len(self.entries)} clips from dataset")

    def load_clip(self, clip_id=0, num_frames=8):
        """Load video clip"""
        if clip_id >= len(self.entries):
            print(f"ERROR: Clip {clip_id} not found (max: {len(self.entries)-1})")
            return False

        entry = self.entries[clip_id]
        self.current_entry = entry

        try:
            shard = self.data_root / entry["shard"]
            with tarfile.open(shard) as tar:
                member = next((m.name for m in tar.getmembers()
                              if entry["match_id"] in m.name and m.name.endswith(".p0.mp4")), None)
                if not member:
                    print("ERROR: Video not found in tar")
                    return False
                data = tar.extractfile(member).read()

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(data)
                path = tmp.name

            try:
                dec = VideoDecoder(path)
                frames = torch.stack([f.cpu() for f in dec])[:num_frames]
                self.current_frames = frames
                return True
            finally:
                try:
                    Path(path).unlink()
                except:
                    pass
        except Exception as e:
            print(f"ERROR: {e}")
            return False

    def show_info(self):
        """Show checkpoint info"""
        print("\n" + "="*60)
        print("World Model Checkpoint-21000")
        print("="*60)
        print(f"Path: {self.ckpt_path.name}")
        print(f"Iteration: {self.ckpt['iter_num']}")
        print(f"Loss diffusion: {self.ckpt['loss_diffusion']:.6f}")
        print(f"State dict: {len(self.ckpt['state_dict'])} parameters")
        print("="*60)

    def show_frames(self):
        """Show current frames info"""
        if self.current_frames is None:
            print("ERROR: No frames loaded")
            return

        print(f"\nLoaded: {self.current_entry['match_id'][:24]}")
        print(f"Shape: {self.current_frames.shape} (T, C, H, W)")
        print(f"\nFrames 0-3: INPUT (context)")
        print(f"Frames 4-7: TARGET (ground truth to predict)")

    def export_frames(self, output_dir="outputs/wm_interactive"):
        """Export current frames"""
        if self.current_frames is None:
            print("ERROR: No frames loaded")
            return False

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i in range(min(8, len(self.current_frames))):
            frame = self.current_frames[i].numpy()
            if frame.ndim == 3 and frame.shape[0] == 3:
                img = Image.fromarray(frame.transpose(1, 2, 0).astype('uint8'))
            elif frame.ndim == 3 and frame.shape[0] == 1:
                img = Image.fromarray(frame[0].astype('uint8'), mode='L')
            else:
                img = Image.fromarray(frame.astype('uint8'))

            label = "INPUT" if i < 4 else "TARGET"
            img.save(output_dir / f"frame_{i:02d}_{label}.png")

        print(f"✓ Exported 8 frames to {output_dir}/")
        return True

    def run_menu(self):
        """Interactive menu"""
        print("\n" + "="*60)
        print("Interactive World Model Player")
        print("="*60)
        self.show_info()
        print("\nCommands:")
        print("  info    - Show checkpoint info")
        print("  load N  - Load clip N (0-9)")
        print("  show    - Show current frames")
        print("  export  - Save frames to disk")
        print("  list    - List available clips")
        print("  quit    - Exit")
        print("="*60)

        while True:
            try:
                cmd = input("\n> ").strip().lower()

                if not cmd:
                    continue
                elif cmd == "quit" or cmd == "exit":
                    print("Goodbye!")
                    break
                elif cmd == "info":
                    self.show_info()
                elif cmd == "show":
                    self.show_frames()
                elif cmd == "export":
                    self.export_frames()
                elif cmd == "list":
                    for i, e in enumerate(self.entries[:10]):
                        print(f"  {i}: {e['match_id'][:24]}")
                    if len(self.entries) > 10:
                        print(f"  ... and {len(self.entries)-10} more")
                elif cmd.startswith("load "):
                    clip_id = int(cmd.split()[1])
                    if self.load_clip(clip_id):
                        print(f"✓ Loaded clip {clip_id}")
                        self.show_frames()
                else:
                    print("Unknown command. Type 'quit' to exit.")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    ckpt_path = r"C:\workspace\world\alakazam-mira-mini\outputs\scratch_ch\checkpoint-56000\checkpoint-21000-horde.pth"
    data_index = Path(r"C:\recordings\mira_wds\train\index.json")

    player = WorldModelPlayer(ckpt_path, data_index)
    player.run_menu()

if __name__ == "__main__":
    main()
