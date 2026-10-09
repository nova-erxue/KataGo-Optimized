#!/usr/bin/env python3
"""Verify iKataGo's selected engine/model by running GTP; no account data is printed."""
import argparse
import json
from pathlib import Path
from integrate_ikatago import find_entry, gtp_check, load_ikatago_config, parse_overrides, resolved_entry, runtime_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New directory for GTP logs and result JSON')
    parser.add_argument('--cuda-root', type=Path)
    parser.add_argument('--library-dir', type=Path, action='append', default=[])
    parser.add_argument('--board-size', type=int, action='append', help='Repeat for multiple sizes; default: 19. Add 9 explicitly if desired. The engine validates supported sizes.')
    parser.add_argument('--load-only', action='store_true', help='Load the model and check GTP without searching')
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args()
    work = args.work.expanduser().resolve()
    katago = load_ikatago_config(work / 'config/conf.yaml')['katago']
    binary = find_entry(katago['bins'], katago['defaultBinName'], 'engine')
    config = find_entry(katago['configs'], katago['defaultConfigName'], 'config')
    model = find_entry(katago['weights'], katago['defaultWeightName'], 'model')
    env = runtime_environment(args.cuda_root, args.library_dir)
    result = gtp_check(resolved_entry(work, binary), resolved_entry(work, model), resolved_entry(work, config),
                       work, env, parse_overrides(binary.get('overrideConfig', '')), args.output.resolve(),
                       boards=[] if args.load_only else (args.board_size or [19]), timeout=args.timeout)
    result.update(engine=binary['name'], model=model['name'], config=config['name'], authenticatedServiceStarted=False)
    (args.output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
