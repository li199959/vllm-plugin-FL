"""Archive reviewable changes after checking captured runtime source hashes."""

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path


def git(root, *args, allow_diff=False):
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, check=False)
    assert result.returncode == 0 or (allow_diff and result.returncode == 1), (
        result.stderr.decode()
    )
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    recorded = json.loads((args.run / "source_hashes.json").read_text())
    for name, expected in recorded.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, (
            f"Changed since launch: {name}"
        )
    tracked = git(root, "diff", "--name-only", "HEAD").decode().splitlines()
    untracked = (
        git(root, "ls-files", "--others", "--exclude-standard").decode().splitlines()
    )
    patch = git(root, "diff", "--binary", "HEAD")
    for name in untracked:
        patch += git(
            root,
            "diff",
            "--no-index",
            "--binary",
            "--",
            "/dev/null",
            name,
            allow_diff=True,
        )
    (args.run / "implementation.patch").write_bytes(patch)
    with tarfile.open(args.run / "implementation_sources.tar.gz", "w:gz") as archive:
        for name in sorted(set(tracked + untracked)):
            if (root / name).is_file():
                archive.add(root / name, arcname=name)
    (args.run / "implementation_version.json").write_text(
        json.dumps(
            {
                "base_commit": git(root, "rev-parse", "HEAD").decode().strip(),
                "worktree": str(root),
                "captured_runtime_source_hashes_verified": True,
                "archive_sha256": hashlib.sha256(
                    (args.run / "implementation_sources.tar.gz").read_bytes()
                ).hexdigest(),
                "patch_sha256": hashlib.sha256(patch).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        "Archived",
        len(tracked) + len(untracked),
        "files; launch source hashes match",
        flush=True,
    )


if __name__ == "__main__":
    main()
