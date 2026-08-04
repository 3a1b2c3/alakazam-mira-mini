#!/usr/bin/env python3
"""Upload codec checkpoint to Horde via SFTP"""

import os
import sys
from pathlib import Path
import paramiko

def upload_checkpoint(host, user, password, local_path, remote_path):
    """Upload file via SFTP with password auth"""

    print(f"Connecting to {host}...")

    # Create SSH client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Connect with password
        ssh.connect(host, username=user, password=password, port=22)
        print("✓ Connected")

        # Open SFTP
        sftp = ssh.open_sftp()

        # Get file size for progress
        local_file = Path(local_path)
        if not local_file.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        file_size = local_file.stat().st_size
        print(f"Uploading {local_path} ({file_size / (1024**3):.2f} GB)...")
        print(f"  → {remote_path}")

        # Upload with progress
        def progress(transferred, total):
            pct = 100 * transferred // total
            print(f"  {pct}% ({transferred}/{total} bytes)", end='\r')

        sftp.put(local_path, remote_path, callback=progress)
        print(f"\n✓ Uploaded to {remote_path}")

        sftp.close()
        ssh.close()
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    import getpass

    host = "10.57.233.24"
    user = "horde"
    local_dir = "outputs"
    remote_dir = "/home/horde/mira/codec_logs"

    files = [
        (f"{local_dir}/codec_ckpt_7000.pth", f"{remote_dir}/checkpoint-7000/checkpoint.pth"),
        (f"{local_dir}/codec_config.yaml", f"{remote_dir}/codec_config.yaml"),
    ]

    password = getpass.getpass(f"Password for {user}@{host}: ")

    print("\nUploading:")
    all_ok = True
    for local_path, remote_path in files:
        if not upload_checkpoint(host, user, password, local_path, remote_path):
            all_ok = False
            print(f"Warning: Failed to upload {Path(local_path).name}")

    if all_ok:
        print(f"\n✓ Uploaded to {remote_dir}/")
        sys.exit(0)
    else:
        print(f"\n✗ Some files failed to upload")
        sys.exit(1)
