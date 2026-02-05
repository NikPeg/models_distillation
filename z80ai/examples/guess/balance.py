#!/usr/bin/env python3
"""
Balance training data by undersampling majority classes.

Usage:
    cat training.txt | ./balance.py | ../../feedme.py
    cat training.txt | ./balance.py --target 2000 > balanced.txt
    cat training.txt | ./balance.py --ratio 1:1:1:0.5 > balanced.txt
"""

import argparse
import random
import sys
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description='Balance training data')
    parser.add_argument('--target', '-t', type=int, default=0,
                        help='Target count per class (0 = use min class size)')
    parser.add_argument('--oversample', '-o', action='store_true',
                        help='Oversample minority classes instead of undersampling majority')
    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--stats', action='store_true',
                        help='Print stats to stderr')
    args = parser.parse_args()

    random.seed(args.seed)

    # Group by answer
    by_answer = defaultdict(list)
    for line in sys.stdin:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '|' not in line:
            continue
        q, a = line.rsplit('|', 1)
        by_answer[a.upper()].append(line)

    if not by_answer:
        print("# No data found", file=sys.stderr)
        return

    # Determine target size
    counts = {k: len(v) for k, v in by_answer.items()}
    min_count = min(counts.values())
    max_count = max(counts.values())

    if args.target > 0:
        target = args.target
    else:
        target = min_count

    if args.stats:
        print(f"# Original distribution:", file=sys.stderr)
        for ans, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"#   {ans}: {cnt:,}", file=sys.stderr)
        print(f"# Target per class: {target:,}", file=sys.stderr)

    # Balance
    balanced = []
    for answer, lines in by_answer.items():
        if args.oversample:
            # Oversample: repeat minority classes
            if len(lines) < target:
                # Repeat with replacement
                sampled = random.choices(lines, k=target)
            else:
                sampled = random.sample(lines, target)
        else:
            # Undersample: take random subset of majority classes
            if len(lines) > target:
                sampled = random.sample(lines, target)
            else:
                sampled = lines  # Keep all if under target
        balanced.extend(sampled)

    # Shuffle
    random.shuffle(balanced)

    if args.stats:
        final_counts = defaultdict(int)
        for line in balanced:
            a = line.rsplit('|', 1)[1].upper()
            final_counts[a] += 1
        print(f"# Balanced distribution:", file=sys.stderr)
        for ans, cnt in sorted(final_counts.items(), key=lambda x: -x[1]):
            print(f"#   {ans}: {cnt:,}", file=sys.stderr)
        print(f"# Total: {len(balanced):,}", file=sys.stderr)

    # Output
    for line in balanced:
        print(line)


if __name__ == '__main__':
    main()
