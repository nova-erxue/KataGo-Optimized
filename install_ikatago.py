#!/usr/bin/env python3
"""Install the packaged engine with the current iKataGo model and a backup."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / 'tools'))
from integrate_ikatago import find_entry, load_ikatago_config, resolved_entry

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--work', required=True, type=Path)
p.add_argument('--model', type=Path, help='Optional replacement for the current default model')
p.add_argument('--cuda-root', type=Path)
p.add_argument('--library-dir', action='append', default=[])
a = p.parse_args()
work = a.work.expanduser().resolve()
try:
    conf = load_ikatago_config(work / 'config/conf.yaml')['katago']
except RuntimeError as exc:
    p.error(str(exc))
weight = find_entry(conf['weights'], conf['defaultWeightName'], 'model')
model = a.model.expanduser().resolve() if a.model else resolved_entry(work, weight)
engine = root / ('katago.exe' if os.name == 'nt' else 'katago')
if not engine.is_file():
    p.error('Use this installer from the extracted binary release, not the source archive.')
cmd = [sys.executable, str(root / 'tools/integrate_ikatago.py'), '--work', str(work),
       '--engine', str(engine), '--model', str(model)]
if not a.model:
    cmd += ['--model-name', weight['name']]
if a.cuda_root:
    cmd += ['--cuda-root', str(a.cuda_root)]
elif os.name == 'nt' and os.environ.get('CUDA_PATH_V13_2'):
    cmd += ['--cuda-root', os.environ['CUDA_PATH_V13_2']]
cmd += ['--library-dir', str(root / 'lib')]
for directory in a.library_dir:
    cmd += ['--library-dir', directory]
for directory in os.environ.get('KATAGO_LIBRARY_PATH', '').split(os.pathsep):
    if directory:
        cmd += ['--library-dir', directory]
sys.exit(subprocess.call(cmd))
