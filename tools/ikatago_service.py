#!/usr/bin/env python3
"""Manage only the iKataGo process belonging to this deployment."""
import argparse
import fcntl
import getpass
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from integrate_ikatago import atomic_bytes, find_entry, gtp_check, load_ikatago_config, runtime_environment

ROOT = Path(__file__).resolve().parents[1]
PID = ROOT/'service.pid.json'

def save(path, data):
    atomic_bytes(path, (json.dumps(data, indent=2)+'\n').encode(), 0o600)

def identity(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z': return None
        return dict(pid=pid, start=stat[19], exe=str(Path(f'/proc/{pid}/exe').resolve(strict=True)))
    except (OSError, ValueError): return None

def active():
    if not PID.exists(): return None
    data = json.loads(PID.read_text())
    return data if identity(data['pid']) == data else None

def environment():
    data = json.loads((ROOT/'config/runtime.json').read_text())
    return runtime_environment(data.get('cudaRoot'), [ROOT/'lib']+data.get('libraryDirectories', []))

def check():
    conf = load_ikatago_config(ROOT/'config/conf.yaml')['katago']
    def entry(kind, default):
        return (ROOT/find_entry(conf[kind], conf[default], kind)['path']).resolve()
    log = ROOT/'logs'/('check-'+str(time.time_ns()))
    gtp_check(entry('bins','defaultBinName'), entry('weights','defaultWeightName'),
              entry('configs','defaultConfigName'), ROOT, environment(), {}, log, boards=[19])
    data = json.loads((ROOT/'deployment.json').read_text())
    data['engineVerified'] = True
    save(ROOT/'deployment.json', data)
    print('GPU/model search check passed. Logs:', log)

def accounts():
    path = ROOT/'userlist.txt'
    if not path.is_file(): raise ValueError('Configure an account first: ./service.sh configure')
    lines = path.read_text().splitlines()
    if not lines: raise ValueError('Account file is empty. Run ./service.sh configure')
    for line in lines:
        parts = line.split(':')
        if len(parts) != 2 or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', parts[0]) or not parts[1] or parts[1].strip()!=parts[1]:
            raise ValueError('Invalid account file. Run ./service.sh configure')
    path.chmod(0o600)

def configure():
    username = input('iKataGo username: ').strip()
    password = getpass.getpass('iKataGo password (hidden): ')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', username): raise ValueError('Invalid username.')
    if not password or any(c in password for c in ':\r\n\x00') or password.strip()!=password:
        raise ValueError('Invalid password: no colon/newline or outer whitespace.')
    if getpass.getpass('Repeat password: ') != password: raise ValueError('Passwords do not match.')
    atomic_bytes(ROOT/'userlist.txt', (username+':'+password+'\n').encode(), 0o600)
    print('Account saved.')

def stop(data):
    if not data: print('Stopped.'); return
    os.kill(data['pid'], signal.SIGTERM)
    for _ in range(50):
        if identity(data['pid']) != data: break
        time.sleep(0.1)
    if identity(data['pid']) == data: os.kill(data['pid'], signal.SIGKILL)
    PID.unlink(missing_ok=True)
    print('Stopped.')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['start','stop','status','foreground','configure','check'])
    action = p.parse_args().action
    os.umask(0o077)
    (ROOT/'logs').mkdir(exist_ok=True, mode=0o700)
    with (ROOT/'service.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = active()
        if action == 'status':
            print('Running (PID '+str(data['pid'])+'); remote connectivity is not verified.' if data else 'Stopped.')
            return 0
        if action == 'stop': stop(data); return 0
        if data: raise ValueError('Service is running. Stop it before checking or configuring.')
        if action == 'configure': configure(); return 0
        if action == 'check': check(); return 0
        accounts()
        if not json.loads((ROOT/'deployment.json').read_text()).get('engineVerified'): check()
        platform = json.loads((ROOT/'config/public-platform.json').read_text())
        argv = [str(ROOT/'ikatago-server'), '--platform', platform['platform'], '--token', platform['token'], '--config', './config/conf.yaml']
        log = ROOT/'logs/server.log'
        with log.open('ab') as output:
            log.chmod(0o600)
            proc = subprocess.Popen(argv, cwd=ROOT, env=environment(), stdin=subprocess.DEVNULL,
                                    stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            time.sleep(1)
            if proc.poll() is not None: raise ValueError('Server exited. Inspect private log: '+str(log))
            data = identity(proc.pid)
            if not data: raise ValueError('Cannot identify server process.')
            save(PID, data)
        except BaseException:
            if proc.poll() is None: proc.terminate(); proc.wait(timeout=10)
            raise
        print('Server process running. Log:', log, flush=True)
        print('Client connection still depends on the upstream iKataGo service.', flush=True)
    if action == 'foreground':
        def terminate(*_):
            if identity(proc.pid) == data: proc.terminate()
        signal.signal(signal.SIGTERM, terminate)
        signal.signal(signal.SIGINT, terminate)
        return proc.wait()
    return 0

if __name__ == '__main__':
    try: sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print('iKataGo: '+str(exc), file=sys.stderr)
        sys.exit(1)
