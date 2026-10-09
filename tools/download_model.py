"""Resolve the official strongest confidently-rated model, then download it."""
import gzip
import hashlib
import html
import os
from pathlib import Path
import re
import tempfile
import urllib.parse
import urllib.request

NETWORKS = 'https://katagotraining.org/networks/'
USER_AGENT = 'KataGo-RTX50-Installer/1.0'

def open_url(url):
    response = urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': USER_AGENT}), timeout=60)
    if urllib.parse.urlsplit(response.url).scheme != 'https':
        response.close()
        raise RuntimeError('Expected an HTTPS download.')
    return response

def strongest():
    with open_url(NETWORKS) as response:
        page = response.read(4*1024*1024+1)
    if len(page) > 4*1024*1024: raise RuntimeError('Network listing exceeds expected size.')
    match = re.search(r'Strongest confidently-rated network:.*?<a\s+[^>]*href=[\"\x27]([^\"\x27]+)', page.decode('utf-8'), re.S)
    if not match: raise RuntimeError('Cannot find the official strongest model. Supply a local model instead.')
    url = html.unescape(match.group(1))
    parts = urllib.parse.urlsplit(url)
    name = Path(parts.path).name
    if parts.scheme != 'https' or parts.hostname != 'media.katagotraining.org' or not re.fullmatch(r'[A-Za-z0-9_.-]+\.bin\.gz', name):
        raise RuntimeError('Unexpected official model link; no model downloaded.')
    return dict(name=name, url=url, selection='official strongest confidently-rated', selectionPage=NETWORKS)

def download(directory):
    selected = strongest()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory/selected['name']
    if target.exists(): raise RuntimeError('Model destination already exists: '+str(target))
    fd, temporary = tempfile.mkstemp(prefix='.model-', dir=directory)
    temporary = Path(temporary)
    digest = hashlib.sha256()
    size = 0
    try:
        print('Downloading official strongest model: '+selected['name'],flush=True)
        with os.fdopen(fd,'wb') as output, open_url(selected['url']) as response:
            expected = response.headers.get('Content-Length')
            while block := response.read(1024*1024):
                size += len(block)
                if size > 2*1024**3: raise RuntimeError('Model exceeds 2 GiB download limit.')
                output.write(block)
                digest.update(block)
            if expected is not None and size != int(expected): raise RuntimeError('Incomplete model download.')
        # Read through the stream to verify gzip CRC, without changing the original weights.
        unpacked = 0
        with gzip.open(temporary,'rb') as compressed:
            while block := compressed.read(1024*1024):
                unpacked += len(block)
                if unpacked > 4*1024**3: raise RuntimeError('Uncompressed model exceeds 4 GiB limit.')
        if unpacked == 0: raise RuntimeError('Empty model.')
        temporary.replace(target)
        selected.update(sha256=digest.hexdigest(),bytes=size)
        return target, selected
    finally:
        temporary.unlink(missing_ok=True)
