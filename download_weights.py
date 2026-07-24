"""Pre-download MIRA Mini world-model weights so `mira-mini play` starts offline.

Downloads the SAME HF repos the package uses (alakazam_mira/weights.py) into the
default Hugging Face cache, so `mira-mini play` finds them ready (bundle_ready)
and skips its own download. hf_xet is disabled because that transfer stalls at
0 bytes on Windows; plain resumable HTTPS is used instead.

Usage:  download_weights.py [1b | 364m | all | <repo_id>]   (default: 1b)
  1b   -> alakazamworld/mira-mini        (discrete GPU, e.g. the RTX 5090)
  364m -> alakazamworld/mira-mini-364m   (laptop / Apple-silicon tier)
"""
import sys

# Uses hf_xet by default (faster chunked transfer) since hf-xet is installed.
# xet stalled on the lingbot-video repo on Windows, so if THIS download hangs at
# 0 bytes, set HF_HUB_DISABLE_XET=1 before running to fall back to plain HTTPS.

from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

REPOS = {
    "1b": "alakazamworld/mira-mini",
    "364m": "alakazamworld/mira-mini-364m",
}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "1b"
    if which == "all":
        repos = list(REPOS.values())
    elif which in REPOS:
        repos = [REPOS[which]]
    else:
        repos = [which]  # allow a raw HF repo id

    failed = []
    for repo in repos:
        print(f"[mira-mini] downloading {repo} (HTTPS, xet disabled) ...", flush=True)
        try:
            path = snapshot_download(repo)
        except (GatedRepoError, RepositoryNotFoundError) as e:
            print(f"[mira-mini] {repo}: not public yet / no access — the weights unlock at launch.")
            print("            If you HAVE access, run `hf auth login` first, then retry.")
            print(f"            ({type(e).__name__})")
            failed.append(repo)
            continue
        except HfHubHTTPError as e:
            code = getattr(getattr(e, "response", None), "status_code", "?")
            print(f"[mira-mini] {repo}: HTTP {code} — {e}")
            failed.append(repo)
            continue
        print(f"[mira-mini] done: {path}")

    if failed:
        print(f"[mira-mini] FAILED: {', '.join(failed)}")
        sys.exit(1)
    print("[mira-mini] all weights cached — run `mira-mini play` to launch.")


if __name__ == "__main__":
    main()
