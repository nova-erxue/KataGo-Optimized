#!/usr/bin/env python3
"""Integrate a rebuilt RTX 50 CUDA engine and any supported KataGo model into iKataGo.

Requires Python 3.9+ and PyYAML. Run on the destination machine. The engine itself
enforces RTX 50 support; no model name, digest, operating system or CUDA path is fixed.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_bytes(path, data, mode=0o644):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
        if os.name != 'nt':
            os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def restore_backup(backup, work=None):
    backup = Path(backup).resolve()
    manifest = json.loads((backup / 'restore-manifest.json').read_text(encoding='utf-8'))
    work = Path(work or manifest['work']).resolve()
    for item in manifest['files']:
        relative = Path(item['relativePath'])
        target, source = work / relative, backup / relative
        if relative.is_absolute() or not contained(target, work) or not contained(source, backup):
            raise ValueError('Backup contains a path outside the work directory')
        if item['existed'] and sha(source) != item['beforeSha256']:
            raise ValueError('Backup file checksum mismatch: ' + str(relative))
    ordered = sorted(manifest['files'], key=lambda item: item['relativePath'].replace('\\', '/') == 'config/conf.yaml')
    for item in ordered:
        target, source = work / item['relativePath'], backup / item['relativePath']
        if item['existed']:
            atomic_bytes(target, source.read_bytes(), source.stat().st_mode & 0o777)
        elif target.exists():
            target.unlink()


def runtime_environment(cuda_root=None, library_dirs=()):
    env = os.environ.copy()
    directories = [str(Path(p).expanduser().resolve()) for p in library_dirs]
    path_prefix = []
    if cuda_root:
        root = Path(cuda_root).expanduser().resolve()
        path_prefix.append(str(root / 'bin'))
        directories.extend(str(p) for p in (root / 'lib64', root / 'lib') if p.is_dir())
    if os.name == 'nt':
        path_prefix.extend(directories)
    elif directories:
        env['LD_LIBRARY_PATH'] = os.pathsep.join(directories + ([env['LD_LIBRARY_PATH']] if env.get('LD_LIBRARY_PATH') else []))
    if path_prefix:
        env['PATH'] = os.pathsep.join(path_prefix + ([env['PATH']] if env.get('PATH') else []))
    return env


def read_settings(path):
    settings = {}
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        line = line.split('#', 1)[0].strip()
        if line:
            key, value = line.split('=', 1)
            settings[key.strip()] = value.strip()
    if not settings:
        raise ValueError('The optimization configuration is empty')
    return settings


def parse_overrides(text):
    return {key.strip(): value.strip() for key, value in
            (item.split('=', 1) for item in str(text or '').split(',') if '=' in item)}


def merged_config(text, settings):
    # Remove every previous definition of a setting, including duplicate definitions.
    lines = []
    for line in text.splitlines():
        match = re.match(r'^\s*([^#=\s]+)\s*=', line)
        if not match or match.group(1) not in settings:
            lines.append(line)
    return '\n'.join(lines) + '\n\n# RTX 50 CUDA settings\n' + ''.join(k + ' = ' + v + '\n' for k, v in settings.items())


def get_yaml():
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError('Install PyYAML first: python -m pip install PyYAML') from exc
    return yaml


def load_ikatago_config(path):
    path = Path(path)
    yaml = get_yaml()
    try:
        content = path.read_text(encoding='utf-8-sig')
    except FileNotFoundError:
        raise RuntimeError('iKataGo configuration missing: '+str(path)+'. For a new installation use deploy-ikatago.sh --model /path/model.bin.gz --work /root/work') from None
    except PermissionError:
        raise RuntimeError('No permission to read iKataGo configuration: '+str(path)) from None
    try:
        config = yaml.safe_load(content)
    except yaml.YAMLError:
        raise RuntimeError('Invalid YAML in '+str(path)+'; private contents were omitted.') from None
    if not isinstance(config, dict) or not isinstance(config.get('katago'), dict):
        raise RuntimeError('Missing katago section in '+str(path))
    return config


def find_entry(entries, name, kind):
    matches = [entry for entry in entries if entry.get('name') == name]
    if len(matches) != 1:
        raise ValueError('Expected exactly one ' + kind + ' entry named ' + str(name))
    return matches[0]


def resolved_entry(work, entry):
    return (work / entry['path']).resolve()


def contained(path, directory):
    try:
        Path(path).resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def gtp_check(engine, model, config, work, env, overrides, output, boards=(), timeout=600):
    output.mkdir(parents=True, exist_ok=False)
    settings = dict(overrides)
    settings.update(maxVisits='16', maxTime='5', ponderingEnabled='false', logToStderr='true',
                    logAllGTPCommunication='false', logSearchInfo='false')
    commands = ['1 protocol_version', '2 name', '3 version']
    expected = {1: r'2', 2: r'KataGo', 3: r'.+'}
    for index, board in enumerate(boards):
        first = 10 + index * 10
        commands += [f'{first} boardsize {board}', f'{first+1} clear_board', f'{first+2} genmove B']
        expected[first] = ''
        expected[first + 1] = ''
        expected[first + 2] = r'(?:[A-HJ-Z][0-9]+|pass|resign)'
    commands.append('99 quit')
    expected[99] = ''
    argv = [str(engine), 'gtp', '-config', str(config), '-model', str(model),
            '-override-config', ','.join(key + '=' + value for key, value in settings.items())]
    try:
        proc = subprocess.run(argv, input='\n'.join(commands) + '\n', text=True,
                              encoding='utf-8', errors='replace', capture_output=True,
                              cwd=work, env=env, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        for name, value in [('gtp.stdout', exc.stdout), ('gtp.stderr', exc.stderr)]:
            if value:
                (output / name).write_bytes(value if isinstance(value, bytes) else value.encode('utf-8'))
        raise RuntimeError('GTP timed out; see ' + str(output)) from exc
    (output / 'gtp.stdout').write_text(proc.stdout, encoding='utf-8')
    (output / 'gtp.stderr').write_text(proc.stderr, encoding='utf-8')
    passed = proc.returncode == 0 and not re.search(r'^\?', proc.stdout, re.M)
    for command_id, pattern in expected.items():
        passed = passed and re.search(r'^=' + str(command_id) + r'[ \t]*' + pattern + r'[ \t]*\r?$', proc.stdout, re.M) is not None
    result = {'passed': bool(passed), 'enginePath': str(engine), 'modelPath': str(model),
              'engineSha256': sha(engine), 'modelSha256': sha(model), 'returnCode': proc.returncode,
              'boardsTested': list(boards), 'loadedModel': bool(passed)}
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    if not passed:
        raise RuntimeError('GTP initialization/search failed; no configuration was installed. See ' + str(output))
    return result


def default_model_name(path):
    name = path.name
    for suffix in ('.gz', '.bin', '.txt'):
        if name.lower().endswith(suffix):
            name = name[:-len(suffix)]
    return name or 'KataGo-model'


def choose_model_entry(work, entries, requested, digest):
    # A collision changes the new name; an existing user's weight is never overwritten.
    for name in [requested, requested + '-' + digest[:12], requested + '-' + digest]:
        existing = [entry for entry in entries if entry.get('name') == name]
        if not existing:
            return name, None
        if len(existing) == 1:
            candidate = resolved_entry(work, existing[0])
            if candidate.is_file() and sha(candidate) == digest:
                return name, existing[0]
    raise ValueError('Model entry name collision; choose another --model-name')


def launcher_bytes(work, cuda_root, library_dirs):
    # This is a separate wrapper. Never read, rewrite or print the account launcher.
    directories = [str(Path(p).expanduser().resolve()) for p in library_dirs]
    path_prefix = []
    if cuda_root:
        root = Path(cuda_root).expanduser().resolve()
        path_prefix.append(str(root / 'bin'))
        directories.extend(str(p) for p in (root / 'lib64', root / 'lib') if p.is_dir())
    if os.name == 'nt':
        def cmd_value(value):
            if any(character in value for character in '\r\n"'):
                raise ValueError('Unsupported quote or newline in Windows runtime path')
            return value.replace('%', '%%')
        lines = ['@echo off', 'setlocal DisableDelayedExpansion', 'cd /d "%~dp0"']
        prefix = ';'.join(path_prefix + directories)
        if prefix:
            lines.append('set "PATH=' + cmd_value(prefix) + ';%PATH%"')
        lines += ['if "%~1"=="" goto default', '%*', 'exit /b %errorlevel%', ':default']
        default = next((name for name in ('run.cmd', 'run.bat') if (work / name).is_file()), None)
        if default:
            lines += ['call "%~dp0' + default + '"', 'exit /b %errorlevel%']
        else:
            lines += ['echo Usage: run-ikatago-rtx50.cmd your-existing-service-command [arguments]', 'exit /b 2']
        return 'run-ikatago-rtx50.cmd', ('\r\n'.join(lines) + '\r\n').encode('utf-8')
    lines = ['#!/bin/sh', 'set -eu', 'cd -- "$(dirname -- "$0")"']
    if path_prefix:
        lines.append('export PATH=' + shlex.quote(':'.join(path_prefix)) + '${PATH:+:$PATH}')
    if directories:
        lines.append('export LD_LIBRARY_PATH=' + shlex.quote(':'.join(directories)) + '${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}')
    lines += ['if [ "$#" -gt 0 ]; then exec "$@"; fi']
    if (work / 'run.sh').is_file():
        lines.append('exec bash ./run.sh')
    else:
        lines += ['echo "Usage: ./run-ikatago-rtx50.sh your-existing-service-command [arguments]" >&2', 'exit 2']
    return 'run-ikatago-rtx50.sh', ('\n'.join(lines) + '\n').encode('utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True, help='Existing iKataGo work directory on this machine')
    parser.add_argument('--engine', type=Path, required=True, help='Native compiled katago executable')
    parser.add_argument('--model', type=Path, required=True, help='Any model supported by this KataGo backend')
    parser.add_argument('--model-name', help='Display name; defaults to the model filename without .bin/.txt/.gz')
    parser.add_argument('--engine-name', default='CUDA-Optimized-RTX50')
    parser.add_argument('--cuda-root', type=Path)
    parser.add_argument('--library-dir', type=Path, action='append', default=[], help='Runtime library or DLL directory; repeat as needed')
    parser.add_argument('--timeout', type=int, default=600, help='GTP initialization timeout in seconds')
    args = parser.parse_args()
    work, engine, model = [path.expanduser().resolve() for path in (args.work, args.engine, args.model)]
    if not work.is_dir() or not engine.is_file() or not model.is_file():
        parser.error('--work, --engine and --model must exist')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', args.engine_name) or args.engine_name in ('.', '..'):
        parser.error('--engine-name must contain only letters, digits, underscore, period or hyphen')
    yaml = get_yaml()
    confpath = work / 'config/conf.yaml'
    conf = load_ikatago_config(confpath)
    katago = conf['katago']
    source_config = find_entry(katago['configs'], katago['defaultConfigName'], 'config')
    settings = read_settings(Path(__file__).parent / 'rtx50-search.cfg')
    cfgtext = merged_config(resolved_entry(work, source_config).read_text(encoding='utf-8-sig'), settings)
    bins = katago.setdefault('bins', [])
    existing = [entry for entry in bins if entry.get('name') == args.engine_name]
    if len(existing) > 1:
        raise ValueError('Duplicate engine names in iKataGo configuration')
    entry = existing[0] if existing else {'name': args.engine_name}
    old_flags = parse_overrides(entry.get('overrideConfig', ''))
    old_flags.update(settings)
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    check_dir = work / ('rtx50-integration-check-' + stamp)
    env = runtime_environment(args.cuda_root, args.library_dir)
    # The engine loads the real model and initializes CUDA before acknowledging GTP.
    with tempfile.TemporaryDirectory(prefix='.rtx50-check-', dir=work) as temporary:
        candidate_config = Path(temporary) / 'candidate.cfg'
        candidate_config.write_text(cfgtext, encoding='utf-8')
        gtp_check(engine, model, candidate_config, work, env, old_flags, check_dir, timeout=args.timeout)

    filename = args.engine_name + ('.exe' if os.name == 'nt' else '')
    target = work / 'data/bins' / filename
    entry.update(path='./data/bins/' + filename, backend='cuda',
                 description='RTX 50 CUDA backend; compatible KataGo TF weights',
                 overrideConfig=','.join(key + '=' + value for key, value in old_flags.items()))
    if not existing:
        bins.insert(0, entry)

    digest = sha(model)
    weights = katago.setdefault('weights', [])
    model_name, weight = choose_model_entry(work, weights, args.model_name or default_model_name(model), digest)
    model_target = None
    if weight is None:
        suffix = ''.join(model.suffixes) or '.bin'
        model_target = work / 'data/weights' / ('rtx50-model-' + digest + suffix)
        weight = {'name': model_name, 'path': './' + model_target.relative_to(work).as_posix(),
                  'description': 'Original model: ' + model.name}
        weights.insert(0, weight)
    cfg_name = args.engine_name + '-config'
    cfgpath = work / 'data/configs' / (args.engine_name + '.cfg')
    config_entries = [item for item in katago['configs'] if item.get('name') == cfg_name]
    if len(config_entries) > 1:
        raise ValueError('Duplicate optimized configuration names')
    cfg_entry = config_entries[0] if config_entries else {'name': cfg_name}
    cfg_entry.update(path='./' + cfgpath.relative_to(work).as_posix(), description='RTX 50 CUDA configuration')
    if not config_entries:
        katago['configs'].insert(0, cfg_entry)
    katago.update(defaultBinName=args.engine_name, defaultWeightName=model_name, defaultConfigName=cfg_name)
    wrapper_name, wrapper = launcher_bytes(work, args.cuda_root, args.library_dir)
    status_path = work / 'rtx50-integration.json'
    writes = [(target, engine.read_bytes(), 0o755), (cfgpath, cfgtext.encode('utf-8'), 0o644),
              (work / wrapper_name, wrapper, 0o755)]
    if model_target is not None and model_target != model:
        writes.append((model_target, model.read_bytes(), 0o644))
    backup = work / ('rtx50-integration-backup-' + stamp)
    backup.mkdir(mode=0o700)
    paths = [row[0] for row in writes] + [confpath, status_path]
    saved_files = []
    for path in paths:
        if not contained(path, work):
            raise ValueError('Refusing to replace a symlink or destination outside --work: ' + str(path))
        relative = path.relative_to(work).as_posix()
        record = {'relativePath': relative, 'existed': path.exists()}
        if path.exists():
            saved = backup / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, saved)
            record['beforeSha256'] = sha(path)
        saved_files.append(record)
    (backup / 'restore-manifest.json').write_text(json.dumps({'work': str(work), 'files': saved_files}, indent=2), encoding='utf-8')
    result = {'backupDirectory': str(backup), 'engineSha256': sha(engine), 'modelSha256': digest,
              'defaultEngine': args.engine_name, 'defaultWeight': model_name, 'defaultConfig': cfg_name,
              'modelLoadVerification': str(check_dir), 'wrapper': str(work / wrapper_name),
              'cudaRoot': str(args.cuda_root.resolve()) if args.cuda_root else None,
              'libraryDirectories': [str(path.resolve()) for path in args.library_dir],
              'serverRestarted': False, 'note': 'Start/restart iKataGo through the wrapper so it inherits the tested library environment.'}
    # A same-name upgrade replaces paths selected by the existing conf.yaml too.
    # Restore all changed files if installation or the installed-path probe fails.
    try:
        for path, data, mode in writes:
            atomic_bytes(path, data, mode)
        installed_check = work / ('rtx50-installed-check-' + stamp)
        gtp_check(target, resolved_entry(work, weight), cfgpath, work, env, old_flags,
                  installed_check, timeout=args.timeout)
        result['installedModelVerification'] = str(installed_check)
        atomic_bytes(status_path, (json.dumps(result, indent=2) + '\n').encode('utf-8'))
        atomic_bytes(confpath, yaml.safe_dump(conf, sort_keys=False, allow_unicode=True).encode('utf-8'), confpath.stat().st_mode & 0o777)
    except BaseException:
        restore_backup(backup, work)
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
