import argparse
from pathlib import Path
from urllib.request import urlretrieve

from orchestrator.config import VoiceConfig


def main() -> None:
    config = VoiceConfig()

    parser = argparse.ArgumentParser(description="Download Silero VAD ONNX model")
    parser.add_argument("--url", default=config.silero_model_url)
    parser.add_argument("--output", default="")
    parser.add_argument("--cache-dir", default=config.silero_model_cache_dir)
    args = parser.parse_args()

    if args.output:
        target = Path(args.output)
    else:
        root_dir = Path(__file__).resolve().parents[2]
        cache_dir = root_dir / args.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / "silero_vad.onnx"

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Silero model from {args.url} to {target}")
    urlretrieve(args.url, target)
    print("Download complete")


if __name__ == "__main__":
    main()
