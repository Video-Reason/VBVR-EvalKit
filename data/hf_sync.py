#!/usr/bin/env python3
"""
HuggingFace Hub Sync Utilities for VMEvalKit

Features:
- Upload a local folder (e.g., data/questions) to a HF dataset repo while preserving structure
- Download a HF dataset repo snapshot to a local folder with the same structure
- List files in a HF dataset repo

Environment:
- Requires HF_TOKEN environment variable for authentication (write or private access)
"""

import os
import argparse
from pathlib import Path
from typing import Optional, List

from huggingface_hub import HfApi, snapshot_download


def _get_token(env_var: str = "HF_TOKEN") -> Optional[str]:
    token = os.getenv(env_var)
    return token


def hf_upload(
    local_path: Path,
    repo_id: str,
    repo_type: str = "dataset",
    path_in_repo: str = "",
    private: bool = False,
    commit_message: Optional[str] = None,
    revision: str = "main",
    token_env: str = "HF_TOKEN",
) -> None:
    token = _get_token(token_env)
    api = HfApi(token=token)

    api.create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=private,
        exist_ok=True,
    )

    message = commit_message or f"Upload {local_path} to {repo_id}:{revision}"
    api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(local_path),
        path_in_repo=path_in_repo,
        revision=revision,
        commit_message=message,
        allow_pattern=None,
        ignore_pattern=None,
    )


def hf_download(
    repo_id: str,
    target_dir: Path,
    repo_type: str = "dataset",
    revision: str = "main",
    token_env: str = "HF_TOKEN",
    local_dir_use_symlinks: bool = False,
) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)

    token = _get_token(token_env)
    out_dir = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        local_dir=str(target_dir),
        local_dir_use_symlinks=local_dir_use_symlinks,
        token=token,
        etag_timeout=10,
    )
    return Path(out_dir)


def hf_list_files(
    repo_id: str,
    repo_type: str = "dataset",
    revision: str = "main",
    token_env: str = "HF_TOKEN",
) -> List[str]:
    token = _get_token(token_env)
    api = HfApi(token=token)
    files = api.list_repo_files(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
    )
    return files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HuggingFace Hub sync utility for VMEvalKit"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    up = sub.add_parser("upload", help="Upload a local folder to a HF dataset repo")
    up.add_argument("--path", required=True, type=Path, help="Local folder path to upload")
    up.add_argument("--repo-id", required=True, type=str, help="HF repo id, e.g. user/vmevalkit-questions")
    up.add_argument("--repo-type", default="dataset", type=str, choices=["dataset", "model", "space"])
    up.add_argument("--path-in-repo", default="", type=str, help="Subdirectory in the repo")
    up.add_argument("--private", action="store_true", help="Create/keep repo private")
    up.add_argument("--commit-message", default=None, type=str, help="Commit message")
    up.add_argument("--revision", default="main", type=str, help="Branch or tag to upload to")
    up.add_argument("--token-env", default="HF_TOKEN", type=str, help="Env var name holding the HF token")

    down = sub.add_parser("download", help="Download a HF dataset repo to a local folder")
    down.add_argument("--repo-id", required=True, type=str, help="HF repo id, e.g. user/vmevalkit-questions")
    down.add_argument("--target", required=True, type=Path, help="Local target directory")
    down.add_argument("--repo-type", default="dataset", type=str, choices=["dataset", "model", "space"])
    down.add_argument("--revision", default="main", type=str, help="Branch or tag to download")
    down.add_argument("--token-env", default="HF_TOKEN", type=str, help="Env var name holding the HF token")
    down.add_argument("--symlinks", action="store_true", help="Use symlinks in local cache (saves space)")

    ls = sub.add_parser("ls", help="List files in a HF repository")
    ls.add_argument("--repo-id", required=True, type=str, help="HF repo id, e.g. user/vmevalkit-questions")
    ls.add_argument("--repo-type", default="dataset", type=str, choices=["dataset", "model", "space"])
    ls.add_argument("--revision", default="main", type=str, help="Branch or tag to list")
    ls.add_argument("--token-env", default="HF_TOKEN", type=str, help="Env var name holding the HF token")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "upload":
        hf_upload(
            local_path=args.path,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            path_in_repo=args.path_in_repo,
            private=args.private,
            commit_message=args.commit_message,
            revision=args.revision,
            token_env=args.token_env,
        )
        print(f"✅ Uploaded {args.path} to {args.repo_id}@{args.revision}")
        return

    if args.command == "download":
        out_dir = hf_download(
            repo_id=args.repo_id,
            target_dir=args.target,
            repo_type=args.repo_type,
            revision=args.revision,
            token_env=args.token_env,
            local_dir_use_symlinks=args.symlinks,
        )
        print(f"✅ Downloaded {args.repo_id}@{args.revision} to {out_dir}")
        return

    if args.command == "ls":
        files = hf_list_files(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            token_env=args.token_env,
        )
        for f in files:
            print(f)
        return


if __name__ == "__main__":
    main()


