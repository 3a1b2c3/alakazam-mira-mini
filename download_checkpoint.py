#!/usr/bin/env python3
"""Download codec checkpoint from Horde via SFTP"""

import os
import sys
from pathlib import Path
import paramiko

def download_checkpoint(host, user, password, remote_path, local_path):
    """Download file via SFTP with password auth"""

    # Create SSH client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Connect with password
        ssh.connect(host, username=user, password=password, port=22)

        # Open SFTP
        sftp = ssh.open_sftp()

        # Create output dir
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)

        # Download with progress
        def progress(transferred, total):
            pct = 100 * transferred // total
            print(f"  {pct}% ({transferred}/{total} bytes)", end='\r')

        print(f"  {Path(remote_path).name}...", end=' ', flush=True)
        sftp.get(remote_path, local_path, callback=progress)
        print("✓")

        sftp.close()
        ssh.close()
        return True

    except Exception as e:
        print(f"✗ {e}")
        return False

if __name__ == "__main__":
    import getpass

    host = "10.57.233.223"
    user = "horde"
    ckpt_dir = "/home/horde/mira/codec_logs"
    local_dir = "outputs"

    print(f"Connecting to {host}...", end=' ', flush=True)
    password = getpass.getpass(f"\nPassword for {user}@{host}: ")

    # Download checkpoint and config
    files = [
        (f"{ckpt_dir}/checkpoint-7000/checkpoint.pth", f"{local_dir}/codec_ckpt_7000.pth"),
        (f"{ckpt_dir}/codec_config.yaml", f"{local_dir}/codec_config.yaml"),
    ]

    print("\nDownloading:")
    all_ok = True
    for remote_path, local_path in files:
        if not download_checkpoint(host, user, password, remote_path, local_path):
            all_ok = False
            print(f"Warning: Failed to download {Path(remote_path).name}")

    if all_ok:
        print(f"\n✓ Ready at {local_dir}/")
        sys.exit(0)
    else:
        print(f"\n✗ Some files failed to download")
        sys.exit(1)
