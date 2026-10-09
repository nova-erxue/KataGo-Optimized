"""Download the unmodified upstream distribution at installation time."""
import base64
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request
import zipfile

URL = 'https://ikatago-resources.oss-cn-beijing.aliyuncs.com/all/linux-work.zip'
SHA256 = '5c82abd42ed8469a6e5d34781a380144a7902a9d2bb296ae15dc074dee458c63'

def fetch(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    archive = destination/'upstream.zip'
    digest = hashlib.sha256()
    size = 0
    try:
        with urllib.request.urlopen(URL, timeout=60) as response, archive.open('wb') as output:
            if not response.url.startswith('https://'): raise RuntimeError('Upstream redirected away from HTTPS.')
            while block := response.read(1024*1024):
                size += len(block)
                if size > 64*1024*1024: raise RuntimeError('Upstream archive exceeds expected size limit.')
                digest.update(block)
                output.write(block)
        if digest.hexdigest() != SHA256:
            raise RuntimeError('Upstream package changed (SHA256 mismatch). Update this installer after reviewing the new upstream release; no downloaded code was executed.')
        with zipfile.ZipFile(archive) as z:
            def read(name):
                info = z.getinfo('linux-work/'+name)
                if info.file_size > 64*1024*1024: raise RuntimeError('Unexpected upstream file size.')
                return z.read(info)
            launcher = read('run.sh').decode()
            tokens = re.findall(r'eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+', launcher)
            if len(tokens) != 1: raise RuntimeError('Unrecognized upstream platform configuration.')
            claims = json.loads(base64.urlsafe_b64decode(tokens[0].split('.')[1]+'==='))
            if claims.get('aud') != 'all' or claims.get('exp',0) <= time.time():
                raise RuntimeError('Upstream public platform token is expired or unsupported.')
            # Only these fixed members are read. The upstream shell script is never executed.
            (destination/'ikatago-server').write_bytes(read('ikatago-server'))
            (destination/'frpc.txt').write_bytes(read('config/frpc.txt'))
            (destination/'public-platform.json').write_text(json.dumps(dict(platform='all',token=tokens[0]))+'\n')
        record = dict(source=URL, archiveSha256=SHA256, serverVersionFromHelp='5.0.1',
                      platformTokenSource='Downloaded public upstream run.sh',
                      files={name:hashlib.sha256((destination/name).read_bytes()).hexdigest()
                             for name in ('ikatago-server','frpc.txt','public-platform.json')})
        (destination/'provenance.json').write_text(json.dumps(record,indent=2)+'\n')
        return record
    finally:
        archive.unlink(missing_ok=True)
