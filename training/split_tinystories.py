import argparse
import hashlib
from pathlib import Path
from typing import Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("standard"))
    parser.add_argument("--train-ratio", type=float, default=0.92)
    parser.add_argument("--valid-ratio", type=float, default=0.04)
    parser.add_argument("--test-ratio", type=float, default=0.04)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize(text: str) -> str:
    return " ".join(text.strip().split())


def hash_fraction(text: str) -> float:
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big")
    return value / (1 << 64)


def load_stories(paths: Iterable[Path]) -> List[str]:
    stories: List[str] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    stories.append(stripped)
    return stories


def allocate(stories: Iterable[str], ratios: Dict[str, float]) -> Dict[str, List[str]]:
    order = ["train", "valid", "test"]
    thresholds: List[tuple[str, float]] = []
    cumulative = 0.0
    for name in order:
        cumulative += ratios[name]
        thresholds.append((name, min(cumulative, 1.0)))
    buckets: Dict[str, List[str]] = {name: [] for name in order}
    seen = set()
    for story in stories:
        key = normalize(story)
        if not key or key in seen:
            continue
        seen.add(key)
        frac = hash_fraction(key)
        for name, threshold in thresholds:
            if frac <= threshold:
                buckets[name].append(story)
                break
    return buckets


def write_outputs(buckets: Dict[str, List[str]], output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "train": output_dir / "TinyStories-train.txt",
        "valid": output_dir / "TinyStories-valid.txt",
        "test": output_dir / "TinyStories-test.txt",
    }
    for name, path in mapping.items():
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output file exists: {path}")
        with path.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(buckets[name]))
            handle.write("\n")


def main() -> None:
    args = parse_args()
    ratios = {
        "train": float(args.train_ratio),
        "valid": float(args.valid_ratio),
        "test": float(args.test_ratio),
    }
    total = sum(ratios.values())
    if not 0.999 <= total <= 1.001:
        raise ValueError("Ratios must sum to 1.0")
    stories = load_stories(args.input)
    if not stories:
        raise ValueError("No stories found")
    buckets = allocate(stories, ratios)
    write_outputs(buckets, args.output_dir, args.overwrite)


if __name__ == "__main__":
    main()
