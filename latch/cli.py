"""CLI for latch.

    latch scan "some text"
    latch scan --file prompt.txt
    latch benchmark
    latch check "text"          # exit 1 if blocked, for shell pipelines
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from latch.corpus import SAMPLES, labels, texts
from latch.detector import metrics, scan, scan_indirect


def _emit(verdict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(verdict.as_dict(), indent=2))
        return
    print(f"{verdict.action.upper():5s}  score {verdict.score}/100")
    if not verdict.signals:
        print("  no injection signals")
        return
    for s in verdict.signals:
        print(f"  [{s.weight:2d}] {s.label} ({s.category})")
        print(f"       why: {s.why}")
        print(f"       at : {s.excerpt}")


def cmd_scan(args) -> int:
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8", errors="replace")
    else:
        text = " ".join(args.text)

    if not text.strip():
        print("nothing to scan", file=sys.stderr)
        return 2

    verdict = scan_indirect(text) if args.indirect else scan(text)
    _emit(verdict, args.json)
    if args.strict:
        return 1 if verdict.action != "allow" else 0
    return 1 if verdict.blocked else 0


def cmd_benchmark(args) -> int:
    predicted = [scan(t).blocked for t in texts()]
    result = metrics(labels(), predicted)

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print("adversarial corpus")
    print(f"  samples   {len(SAMPLES)}  ({sum(1 for s in SAMPLES if s[1])} attacks, "
          f"{sum(1 for s in SAMPLES if not s[1])} benign)")
    print(f"  precision {result['precision']:.3f}")
    print(f"  recall    {result['recall']:.3f}")
    print(f"  f1        {result['f1']:.3f}")
    print(f"  confusion tp={result['tp']} fp={result['fp']} fn={result['fn']} tn={result['tn']}")

    if args.verbose:
        print("\nmissed attacks:")
        for text, is_attack, note in SAMPLES:
            v = scan(text)
            if is_attack and not v.blocked:
                print(f"  [{v.score:3d}] {note}")
        print("\nfalse positives:")
        for text, is_attack, note in SAMPLES:
            v = scan(text)
            if not is_attack and v.blocked:
                print(f"  [{v.score:3d}] {note}")

    # Fail the run if the benchmark regresses, so this is usable in CI.
    if args.min_precision is not None and result["precision"] < args.min_precision:
        print(f"\nFAIL: precision {result['precision']} < {args.min_precision}", file=sys.stderr)
        return 1
    if args.min_recall is not None and result["recall"] < args.min_recall:
        print(f"\nFAIL: recall {result['recall']} < {args.min_recall}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="latch",
        description="Detect prompt injection before it reaches the model.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="scan a string or file")
    s.add_argument("text", nargs="*", help="text to scan")
    s.add_argument("--file", help="read the text from a file instead")
    s.add_argument("--indirect", action="store_true",
                   help="treat the text as retrieved content (stricter)")
    s.add_argument("--json", action="store_true", help="machine-readable output")
    s.add_argument("--strict", action="store_true",
                   help="exit 1 on flag as well as block")
    s.set_defaults(func=cmd_scan)

    b = sub.add_parser("benchmark", help="run the adversarial corpus")
    b.add_argument("--json", action="store_true")
    b.add_argument("--verbose", action="store_true", help="list misses and false positives")
    b.add_argument("--min-precision", type=float, default=None)
    b.add_argument("--min-recall", type=float, default=None)
    b.set_defaults(func=cmd_benchmark)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
