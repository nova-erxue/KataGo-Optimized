#!/usr/bin/env python3
"""Install the packaged engine and original model, without installing iKataGo."""
import argparse
import json
from pathlib import Path
import platform
import shutil
import sys
import tempfile

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tools'))
from integrate_ikatago import gtp_check, runtime_environment, sha
from download_model import download

def prepare(target, model=None):
    if sys.platform!='linux' or platform.machine()!='x86_64':
        raise RuntimeError('Requires Linux x86_64.')
    target=Path(target).expanduser().absolute()
    if target.is_symlink(): raise RuntimeError('Destination must not be a symlink.')
    metadata=json.loads((ROOT/'BINARY_METADATA.json').read_text())
    if sha(ROOT/'katago')!=metadata['linux']['sha256']: raise RuntimeError('Packaged engine checksum mismatch.')
    if sha(ROOT/'lib/libzip.so.4')!=metadata['libzip']['librarySha256']: raise RuntimeError('Packaged libzip checksum mismatch.')
    if target.exists() and any(target.iterdir()):
        saved=target/'prepared.json'
        if not saved.is_file(): raise RuntimeError('Destination is not empty. Choose a new --destination.')
        old=json.loads(saved.read_text())
        if old.get('engineVerified') is not True: raise RuntimeError('Existing installation has not passed its GPU check.')
        path=(target/old['model']).resolve()
        path.relative_to(target.resolve())
        if sha(target/'katago')!=metadata['linux']['sha256'] or sha(path)!=old['modelSha256']:
            raise RuntimeError('Existing engine/model changed. Choose a new installation directory.')
        if model and sha(Path(model).expanduser().resolve())!=old['modelSha256']:
            raise RuntimeError('A different model is already installed. Choose a new installation directory.')
        print('Keeping installed engine/model:',path)
        print('To select a new strongest model, use a new --destination directory.')
        return old
    target.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+target.name+'-prepare-',dir=target.parent))
    try:
        for name in ('katago','run-katago.sh','default.cfg','LICENSE','BINARY_METADATA.json','deploy_ikatago.py','deploy-ikatago.sh'):
            shutil.copy2(ROOT/name,stage/name)
        for name in ('tools','licenses','third_party','lib'):
            shutil.copytree(ROOT/name,stage/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for name in ('katago','run-katago.sh','deploy-ikatago.sh'): (stage/name).chmod(0o755)
        (stage/'models').mkdir()
        if model:
            original=Path(model).expanduser().resolve()
            selected=stage/'models'/original.name
            shutil.copy2(original,selected)
            source=dict(selection='user supplied local model',name=original.name,sha256=sha(selected))
        else:
            selected,source=download(stage/'models')
        cuda='/usr/local/cuda-13.2' if Path('/usr/local/cuda-13.2').is_dir() else None
        env=runtime_environment(cuda,[stage/'lib'])
        gtp_check(stage/'katago',selected,stage/'default.cfg',stage,env,{},stage/'logs/prepare-gtp',boards=[19])
        record=dict(model=selected.relative_to(stage).as_posix(),modelSha256=sha(selected),
                    engineSha256=metadata['linux']['sha256'],modelSource=source,engineVerified=True)
        (stage/'prepared.json').write_text(json.dumps(record,indent=2)+'\n')
        if target.exists(): target.rmdir()
        stage.rename(target)
        print('KataGo ready:',target)
        return record
    finally:
        if stage.exists():
            if (stage/'logs').exists():
                logs=target.parent/(stage.name+'-logs')
                (stage/'logs').rename(logs)
                print('Failure logs:',logs,file=sys.stderr)
            shutil.rmtree(stage)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination',type=Path,default=Path.home()/'katago-rtx50')
    p.add_argument('--model',type=Path,help='Optional local weight; default downloads official strongest confidently-rated model')
    args=p.parse_args()
    try: prepare(args.destination,args.model)
    except (RuntimeError,ValueError,OSError) as exc:
        print('KataGo installation failed: '+str(exc),file=sys.stderr);sys.exit(1)
