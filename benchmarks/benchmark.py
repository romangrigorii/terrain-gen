"""Reproducible wall-clock stage benchmark with optional JIT warmup."""
import argparse
import statistics
import time

from terrain.world import Config, generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', nargs=2, type=int, default=[384, 288], metavar=('W', 'H'))
    ap.add_argument('--repeat', type=int, default=3)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()
    for backend in ('numpy', 'numba'):
        try:
            t0 = time.perf_counter()
            generate(Config(width=64, height=48, seed=args.seed, noise_backend=backend), verbose=False)
            cold = time.perf_counter() - t0
        except ImportError as exc:
            print(f'{backend}: SKIPPED: {exc}')
            continue
        times = []
        stages = []
        for i in range(args.repeat):
            t0 = time.perf_counter()
            w = generate(Config(width=args.size[0], height=args.size[1],
                                seed=args.seed + i, noise_backend=backend), verbose=False)
            times.append(time.perf_counter() - t0)
            stages.append(w.timings)
        print(f'{backend}: cold startup (64x48)={cold:.3f}s, '
              f'world median={statistics.median(times):.3f}s '
              f'({args.size[0]}x{args.size[1]}; seeds {args.seed}..{args.seed+args.repeat-1})')
        for name in stages[0]:
            print(f'  {name:12s}: {statistics.median(s[name] for s in stages):.3f}s')

if __name__ == '__main__':
    main()
