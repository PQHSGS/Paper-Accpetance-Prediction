import sys

from DataPipeline.config import Config
from DataPipeline.feature_pipeline import FeaturePipeline


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"Usage: {argv[0]} <config.json>")
        return 1

    cfg = Config(argv[1])
    pipeline = FeaturePipeline(config=cfg)
    pipeline.run(extract_features=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
