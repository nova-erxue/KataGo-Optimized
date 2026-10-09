#!/usr/bin/env python3
"""Create a complete Linux iKataGo installation; never overwrite existing work."""
import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'tools'))
from integrate_ikatago import gtp_check, runtime_environment
from download_ikatago import fetch

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def write(path, content, mode=0o644):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8', newline='\n')
    path.chmod(mode)

def credentials(username):
    username = username or input('iKataGo username: ').strip()
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', username):
        raise ValueError('Username: use letters, numbers, _ or -, up to 64 characters.')
    password = getpass.getpass('iKataGo password (hidden): ')
    if not password or any(c in password for c in ':\r\n\x00') or password.strip() != password:
        raise ValueError('Password must be nonempty, without colon/newline or outer whitespace.')
    if getpass.getpass('Repeat password: ') != password:
        raise ValueError('Passwords do not match.')
    return username + ':' + password + '\n'

def deploy(args):
    if sys.platform != 'linux' or platform.machine() != 'x86_64':
        raise ValueError('This full server deployment package is for Linux x86_64.')
    try: import yaml
    except ImportError: raise ValueError('PyYAML missing. Run setup-linux.sh or: python3 -m pip install PyYAML') from None
    target = args.work.expanduser().absolute()
    if target.is_symlink(): raise ValueError('--work must not be a symlink.')
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError('Target is not empty. For an existing iKataGo use install_ikatago.py; otherwise choose a new --work directory.')
    model = args.model.expanduser().resolve()
    if not model.is_file(): raise ValueError('Model not found. Supply the original TF model with --model.')
    if not 1 <= args.port <= 65535: raise ValueError('Port must be 1..65535.')
    engine = ROOT/'katago'
    if not engine.is_file(): raise ValueError('Use the Linux binary release; the source archive does not contain katago.')
    metadata = json.loads((ROOT/'BINARY_METADATA.json').read_text())
    if sha(engine) != metadata['linux']['sha256']: raise ValueError('KataGo binary checksum mismatch.')
    if sha(ROOT/'lib/libzip.so.4') != metadata['libzip']['librarySha256']: raise ValueError('libzip checksum mismatch.')
    cuda = str(args.cuda_root.resolve()) if args.cuda_root else ('/usr/local/cuda-13.2' if Path('/usr/local/cuda-13.2').is_dir() else None)
    libraries = [str(Path(p).expanduser().resolve()) for p in args.library_dir]
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.'+target.name+'-deploy-', dir=target.parent))
    committed = False
    try:
        assets = stage/'.upstream-download'
        print('Downloading iKataGo directly from the upstream server...',flush=True)
        fetch(assets)
        for d in ('config','data/bins','data/configs','data/weights','lib','tools','user-data','logs'):
            (stage/d).mkdir(parents=True,exist_ok=True)
        (stage/'logs').chmod(0o700)
        shutil.copy2(engine, stage/'data/bins/CUDA-Optimized-RTX50')
        (stage/'data/bins/CUDA-Optimized-RTX50').chmod(0o755)
        shutil.copy2(assets/'ikatago-server',stage/'ikatago-server')
        (stage/'ikatago-server').chmod(0o755)
        shutil.copy2(ROOT/'lib/libzip.so.4',stage/'lib/libzip.so.4')
        model_rel = 'data/weights/'+model.name
        shutil.copy2(model,stage/model_rel)
        shutil.copy2(ROOT/'default.cfg',stage/'data/configs/default_gtp.cfg')
        for name in ('integrate_ikatago.py','verify_ikatago.py','ikatago_service.py'):
            shutil.copy2(ROOT/'tools'/name,stage/'tools'/name)
        shutil.copytree(ROOT/'licenses',stage/'licenses',dirs_exist_ok=True)
        shutil.copy2(ROOT/'LICENSE',stage/'LICENSE')
        shutil.copy2(ROOT/'third_party/ikatago/NOTICE.md',stage/'licenses/ikatago-NOTICE.md')
        shutil.copy2(assets/'provenance.json',stage/'config/ikatago-provenance.json')
        shutil.copy2(assets/'public-platform.json',stage/'config/public-platform.json')
        frp=(assets/'frpc.txt').read_text().replace('local_port = 2223','local_port = '+str(args.port))
        write(stage/'config/frpc.txt',frp)
        conf = dict(server=dict(listen='0.0.0.0:'+str(args.port)), katago=dict(
            bins=[dict(name='CUDA-Optimized-RTX50',path='./data/bins/CUDA-Optimized-RTX50',backend='cuda')],
            weights=[dict(name='TF-model',path='./'+model_rel)],
            configs=[dict(name='RTX50',path='./data/configs/default_gtp.cfg')],
            defaultBinName='CUDA-Optimized-RTX50', defaultWeightName='TF-model', defaultConfigName='RTX50',
            customConfigDir='./user-data',enableWeightsDetectionInDir='./data/weights'),
            use_nat='frp', nats=dict(frp=dict(type='frp',config_file='./config/frpc.txt')),users=dict(file='./userlist.txt'))
        write(stage/'config/conf.yaml',yaml.safe_dump(conf,sort_keys=False))
        runtime=dict(cudaRoot=cuda,libraryDirectories=libraries)
        write(stage/'config/runtime.json',json.dumps(runtime,indent=2)+'\n')
        launcher='#!/bin/sh\nset -eu\ncd -- "$(dirname -- "$0")"\nexec '+shlex.quote(sys.executable)+' ./tools/ikatago_service.py "$@"\n'
        write(stage/'service.sh',launcher,0o755)
        write(stage/'run.sh','#!/bin/sh\nset -eu\ncd -- "$(dirname -- "$0")"\nexec ./service.sh foreground\n',0o755)
        write(stage/'run-ikatago-rtx50.sh','#!/bin/sh\nset -eu\ncd -- "$(dirname -- "$0")"\nexec ./service.sh foreground\n',0o755)
        env=runtime_environment(cuda,[stage/'lib']+libraries)
        help_result=subprocess.run([str(stage/'ikatago-server'),'--help'],capture_output=True,text=True,timeout=20)
        if 'Server Version: 5.0.1' not in help_result.stdout or '--platform' not in help_result.stdout:
            raise ValueError('Downloaded iKataGo server cannot execute on this system.')
        if not args.skip_engine_check:
            gtp_check(stage/'data/bins/CUDA-Optimized-RTX50',stage/model_rel,stage/'data/configs/default_gtp.cfg',
                      stage,env,{},stage/'logs/deploy-gtp',boards=[19],timeout=600)
        if not args.no_account:
            write(stage/'userlist.txt',credentials(args.username),0o600)
        status=dict(engineVerified=not args.skip_engine_check,modelSha256=sha(model),
                    engineSha256=sha(engine),serverVersion='5.0.1',serverStarted=False)
        write(stage/'deployment.json',json.dumps(status,indent=2)+'\n')
        shutil.rmtree(assets)
        if target.exists(): target.rmdir() # Only the empty directory checked above; fails if changed.
        stage.rename(target)
        committed=True
        print('Installed:',target)
        if args.no_account: print('Configure account: '+str(target/'service.sh')+' configure')
        if args.skip_engine_check: print('GPU check was skipped. Run: '+str(target/'service.sh')+' check')
        print('Start: '+str(target/'service.sh')+' start')
        print('Status: '+str(target/'service.sh')+' status')
        print('Stop: '+str(target/'service.sh')+' stop')
    finally:
        if not committed and stage.is_dir() and stage.parent == target.parent:
            if (stage/'logs').is_dir() and any((stage/'logs').rglob('*')):
                failure_logs = target.parent/(stage.name+'-logs')
                (stage/'logs').rename(failure_logs)
                failure_logs.chmod(0o700)
                print('Failure logs preserved: '+str(failure_logs),file=sys.stderr)
            shutil.rmtree(stage)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,default=Path.home()/'ikatago-work')
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--username')
    p.add_argument('--port',type=int,default=2223)
    p.add_argument('--cuda-root',type=Path)
    p.add_argument('--library-dir',action='append',default=[])
    p.add_argument('--no-account',action='store_true',help='Configure later with service.sh configure')
    p.add_argument('--skip-engine-check',action='store_true',help='Stage files before GPU setup; start remains blocked until check passes')
    args=p.parse_args()
    try: deploy(args)
    except (ValueError,RuntimeError,OSError,subprocess.SubprocessError) as exc:
        print('Deployment failed: '+str(exc),file=sys.stderr); return 1
    return 0
if __name__=='__main__': sys.exit(main())
