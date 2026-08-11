"""CLI: predict skin lesion classes for one or more images.

    python -m ai.inference.predict IMAGE [IMAGE ...]
        --checkpoint PATH          required; never guessed
        [--data-config PATH]       default ai/configs/dataset.yaml
        [--device {auto,cpu,cuda,mps}]
        [--top-k N]                default 3
        [--thresholds PATH]        opt-in operating point
        [--require-manifest-check] fail if the val manifest cannot be checked
        [--output PATH]            default: JSON on stdout

`--checkpoint` is required rather than defaulted. Inference is the point where
a wrong set of weights produces confident, plausible, wrong answers, so which
checkpoint ran is always something the caller stated.

Machine-readable JSON goes to stdout and the human summary to stderr, so the
output stays pipeable.

Exit codes: 0 success, 1 the prediction was refused (bad image, bad checkpoint,
failed provenance check), 2 the run itself failed.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ai.inference.images import InvalidImageError
from ai.inference.predictor import DEFAULT_TOP_K, Predictor
from ai.preprocessing.config import DEFAULT_CONFIG_PATH as DEFAULT_DATA_CONFIG_PATH
from ai.preprocessing.config import load_config
from ai.preprocessing.labels import NUM_CLASSES
from ai.training.thresholds import ThresholdSet

EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_ERROR = 2

# Refusals: the inputs or their provenance were unacceptable. Anything else is
# a bug or an environment failure, and must stay distinguishable from these.
REFUSAL_ERRORS = (FileNotFoundError, InvalidImageError, ValueError)


def run(
    images: list[Path],
    checkpoint: Path,
    data_config_path: Path,
    device: str = "auto",
    top_k: int = DEFAULT_TOP_K,
    thresholds_path: Path | None = None,
    require_manifest: bool = False,
) -> dict[str, Any]:
    """Predict for `images`, returning the JSON-ready payload."""
    if not 1 <= top_k <= NUM_CLASSES:
        raise ValueError(f"--top-k must be in [1, {NUM_CLASSES}], got {top_k}")

    data_config = load_config(data_config_path)
    predictor = Predictor(checkpoint, data_config, device=device)

    # Explicit only: no directory is searched for an operating point.
    thresholds = None
    if thresholds_path is not None:
        thresholds = ThresholdSet.load(thresholds_path)

    predictions = predictor.predict_batch(
        list(images),
        top_k=top_k,
        thresholds=thresholds,
        require_manifest=require_manifest,
    )
    return {
        "device": str(predictor.device),
        "thresholds": None if thresholds_path is None else str(thresholds_path),
        "results": [prediction.to_dict() for prediction in predictions],
    }


def _print_summary(payload: dict[str, Any]) -> None:
    """Human-readable summary, on stderr so stdout stays clean JSON."""
    print(f"device: {payload['device']}", file=sys.stderr)
    for result in payload["results"]:
        print(
            f"{result['source']}: {result['predicted_code']} "
            f"({result['predicted_name']}) confidence "
            f"{result['confidence']:.4f}",
            file=sys.stderr,
        )
        threshold = result.get("threshold_prediction")
        if threshold is None:
            continue
        decision = (
            "abstained"
            if threshold["abstained"]
            else f"{threshold['predicted_code']} ({threshold['predicted_name']})"
        )
        verification = threshold["verification"]
        print(
            f"    operating point: {decision}; manifest_verified="
            f"{str(verification['manifest_verified']).lower()}",
            file=sys.stderr,
        )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="checkpoint to run; required so the weights are never guessed",
    )
    parser.add_argument(
        "--data-config", type=Path, default=DEFAULT_DATA_CONFIG_PATH
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=None,
        help="apply a val-fitted operating point from this file",
    )
    parser.add_argument(
        "--require-manifest-check",
        action="store_true",
        help="fail unless the validation manifest fingerprint can be verified",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    try:
        payload = run(
            args.images,
            args.checkpoint,
            args.data_config,
            device=args.device,
            top_k=args.top_k,
            thresholds_path=args.thresholds,
            require_manifest=args.require_manifest_check,
        )
    except REFUSAL_ERRORS as error:
        print(f"prediction refused: {error}", file=sys.stderr)
        raise SystemExit(EXIT_REFUSED) from error
    except Exception as error:  # noqa: BLE001 - the CLI boundary
        print(f"inference failed: {error}", file=sys.stderr)
        raise SystemExit(EXIT_ERROR) from error

    document = json.dumps(payload, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(document + "\n", encoding="utf-8")
    else:
        print(document)

    _print_summary(payload)
    raise SystemExit(EXIT_OK)


if __name__ == "__main__":
    main()
