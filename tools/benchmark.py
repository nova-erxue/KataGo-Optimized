#!/usr/bin/env python3
"""Run one reproducible NN or search benchmark with any supported KataGo model.

The optimized profile reads the adjacent rtx50-search.cfg; CLI choices for batch,
search threads and NN server threads always take precedence. No model whitelist.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from integrate_ikatago import merged_config, read_settings, runtime_environment, sha


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def parse_search(text, threads, positions):
    number = r'([0-9]+(?:\.[0-9]+)?)'
    pattern = (r'numSearchThreads\s*=\s*(\d+):\s*(\d+)\s*/\s*(\d+) positions,'
               r'\s*visits/s\s*=\s*' + number + r'\s*nnEvals/s\s*=\s*' + number +
               r'\s*nnBatches/s\s*=\s*' + number + r'\s*avgBatchSize\s*=\s*' + number)
    rows = [row for row in re.findall(pattern, text)
            if int(row[0]) == threads and int(row[1]) == positions and int(row[2]) == positions]
    if not rows:
        raise ValueError('No completed search result for the requested threads/positions; inspect stdout.log')
    row = rows[-1]
    return {'searchThreads': int(row[0]), 'positions': int(row[1]),
            'visitsPerSecond': float(row[3]), 'nnEvalsPerSecond': float(row[4]),
            'nnBatchesPerSecond': float(row[5]), 'averageBatchSize': float(row[6])}


def command(args, config):
    argv = [str(args.engine), 'benchmarknn' if args.mode == 'nn' else 'benchmark',
            '-model', str(args.model), '-config', str(config)]
    if args.mode == 'nn':
        argv += ['-batch-size', str(args.batch_size), '-boardsize', '19',
                 '-iterations', str(args.iterations), '-warmup', str(args.warmup),
                 '-require-exact-nnlen', '-json']
    else:
        argv += ['-threads', str(args.threads), '-visits', str(args.visits),
                 '-numpositions', str(args.positions), '-boardsize', '19',
                 '-fixed-batch-size', str(args.batch_size),
                 '-no-server-thread-test', '-no-half-batch-size-test']
    return argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path, help='Shared base GTP config; use bundled benchmark-baseline.cfg')
    parser.add_argument('--output-dir', required=True, type=Path, help='New directory; existing results are never overwritten')
    parser.add_argument('--mode', choices=('search', 'nn'), default='search')
    parser.add_argument('--profile', choices=('optimized', 'baseline'), default='optimized',
                        help='baseline leaves optimization settings out; use with the official control engine')
    parser.add_argument('--threads', type=positive, default=48)
    parser.add_argument('--batch-size', type=positive, default=16)
    parser.add_argument('--server-threads', type=positive, default=2)
    parser.add_argument('--visits', type=positive, default=10000)
    parser.add_argument('--positions', type=positive, default=12)
    parser.add_argument('--iterations', type=positive, default=600)
    parser.add_argument('--warmup', type=int, default=60)
    parser.add_argument('--timeout', type=positive, default=3600, help='Maximum whole process duration, including initialization, in seconds')
    parser.add_argument('--cuda-root', type=Path)
    parser.add_argument('--library-dir', type=Path, action='append', default=[])
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error('--warmup must be nonnegative')
    for key in ('engine', 'model', 'config'):
        value = getattr(args, key).expanduser().resolve()
        if not value.is_file():
            parser.error('--' + key + ' file does not exist: ' + str(value))
        setattr(args, key, value)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        parser.error('--output-dir already exists; choose a new path')
    optimization_file = Path(__file__).resolve().parent / 'rtx50-search.cfg'
    optimization = read_settings(optimization_file)
    base_settings = read_settings(args.config)
    if args.profile == 'baseline':
        # Reject accidental comparison against an already optimized GTP config.
        special_keys = set(optimization) - {'numSearchThreads', 'nnMaxBatchSize',
                                            'numNNServerThreadsPerModel', 'cudaUseFP16', 'cudaUseNHWC'}
        present = sorted(set(base_settings) & special_keys)
        if present:
            parser.error('Baseline config contains optimization keys: ' + ', '.join(present) +
                         '. Use the bundled benchmark-baseline.cfg for both engines.')
    settings = dict(optimization) if args.profile == 'optimized' else {}
    settings.update(numSearchThreads=str(args.threads), nnMaxBatchSize=str(args.batch_size),
                    numNNServerThreadsPerModel=str(args.server_threads), cudaUseFP16='true', cudaUseNHWC='true',
                    nnRandSeed='katago-opt')
    effective = merged_config(args.config.read_text(encoding='utf-8-sig'), settings)
    output.mkdir(parents=True)
    config = output / 'effective.cfg'
    config.write_text(effective, encoding='utf-8')
    argv = command(args, config)
    env = runtime_environment(args.cuda_root, args.library_dir)
    record = {'startedUtc': datetime.now(timezone.utc).isoformat(), 'mode': args.mode,
              'profile': args.profile, 'argv': argv, 'engine': str(args.engine),
              'engineSha256': sha(args.engine), 'model': str(args.model), 'modelSha256': sha(args.model),
              'sourceConfig': str(args.config), 'sourceConfigSha256': sha(args.config),
              'effectiveConfigSha256': sha(config), 'optimizationConfigSha256': sha(optimization_file),
              'searchThreads': args.threads, 'batchSize': args.batch_size,
              'serverThreads': args.server_threads, 'boardSize': 19,
              'cudaRoot': str(args.cuda_root.resolve()) if args.cuda_root else None,
              'libraryDirectories': [str(path.resolve()) for path in args.library_dir],
              'passed': False}
    manifest = output / 'result.json'
    manifest.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print('Running ' + args.mode + ' benchmark. Logs: ' + str(output), flush=True)
    start = time.monotonic()
    error = None
    try:
        with (output / 'stdout.log').open('wb') as stdout, (output / 'stderr.log').open('wb') as stderr:
            proc = subprocess.run(argv, cwd=output, env=env, stdout=stdout, stderr=stderr, timeout=args.timeout)
        record['returnCode'] = proc.returncode
        if proc.returncode != 0:
            raise RuntimeError('Engine exited with code ' + str(proc.returncode))
        text = (output / 'stdout.log').read_text(encoding='utf-8', errors='replace')
        if args.mode == 'nn':
            result = json.loads(text)
            if (result.get('batchSize') != args.batch_size or result.get('numThreads') != args.server_threads
                    or result.get('usingFP16') is not True or result.get('boardSizes') != [19]
                    or result.get('requireExactNNLen') is not True):
                raise ValueError('NN benchmark effective settings do not match the requested FP16/board/batch/server settings')
        else:
            result = parse_search(text, args.threads, args.positions)
        record.update(passed=True, measurements=result)
    except (OSError, subprocess.TimeoutExpired, ValueError, RuntimeError) as exc:
        error = str(exc)
        record['error'] = error
    finally:
        record.update(completedUtc=datetime.now(timezone.utc).isoformat(), processSeconds=time.monotonic() - start)
        manifest.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    if error:
        print('Benchmark failed: ' + error + '. See ' + str(output), file=sys.stderr)
        return 1
    print('Benchmark completed: ' + str(manifest))
    if args.mode == 'nn':
        measured = record['measurements']
        print('NN eval/s: ' + str(measured.get('actualWallNNEvalsPerSec')))
        print('Per-server mean batch latency (ms): ' + str(measured.get('perThreadMeanMs')))
    else:
        print('Visits/s: ' + str(record['measurements']['visitsPerSecond']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
