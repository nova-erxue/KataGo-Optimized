#!/usr/bin/env python3
"""Restore only recorded iKataGo integration files from a verified local backup."""
import argparse
import json
from pathlib import Path
from integrate_ikatago import restore_backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('backup', type=Path)
    parser.add_argument('--work', type=Path, help='Override original work directory after moving a backup')
    args = parser.parse_args()
    backup = args.backup.expanduser().resolve()
    restore_backup(backup, args.work)
    print('Restored the recorded iKataGo files. Restart iKataGo to load them.')


if __name__ == '__main__':
    main()
