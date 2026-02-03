#!/usr/bin/env python
"""
Download utility for 3RScan public data release.

The data is released under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 License.

Each scan is identified by a unique ID listed here:
- http://campar.in.tum.de/public_datasets/3RScan/scans.txt
- https://github.com/WaldJohannaU/3RScan/tree/master/splits

Usage:
    # Download the entire 3RScan release:
    python download_3rscan.py -o /path/to/output

    # Download a specific scan:
    python download_3rscan.py -o /path/to/output --id 19eda6f4-55aa-29a0-8893-8eac3a4d8193

    # Download tfrecords:
    python download_3rscan.py -o /path/to/output --type=tfrecords

Related resources:
    - Metadata: http://campar.in.tum.de/public_datasets/3RScan/3RScan.json
    - 3D semantic scene graphs: https://3dssg.github.io
"""
import argparse
import os
import re
import sys
import tempfile
from typing import List, Optional

if sys.version_info.major >= 3 and sys.version_info.minor >= 6:
    import urllib.request as urllib
else:
    import urllib  # type: ignore


BASE_URL = 'http://campar.in.tum.de/public_datasets/3RScan/'
DATA_URL = BASE_URL + 'Dataset/'
TOS_URL = 'http://campar.in.tum.de/public_datasets/3RScan/3RScanTOU.pdf'

TEST_FILETYPES = [
    'mesh.refined.v2.obj',
    'mesh.refined.mtl',
    'mesh.refined_0.png',
    'sequence.zip'
]

# Semantic annotations available only for train/validation scans
# and reference scans in the test set
FILETYPES = TEST_FILETYPES + [
    'labels.instances.annotated.v2.ply',
    'mesh.refined.0.010000.segs.v2.json',
    'semseg.v2.json'
]

RELEASE = 'release_scans.txt'
HIDDEN_RELEASE = 'test_rescans.txt'
RELEASE_SIZE = '~94GB'

id_reg = re.compile(r"[a-z0-9]{8}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{12}")


def get_scans(scan_file: str) -> List[str]:
    """
    Fetch scan IDs from a remote scan list file.

    Args:
        scan_file: URL to the scan list file.

    Returns:
        List of scan UUIDs found in the file.
    """
    scan_lines = urllib.urlopen(scan_file)
    scans = []
    for scan_line in scan_lines:
        scan_line = scan_line.decode('utf8').rstrip('\n')
        match = id_reg.search(scan_line)
        if match:
            scan_id = match.group()
            scans.append(scan_id)
    return scans


def download_release(release_scans: List[str], out_dir: str, file_types: List[str]) -> None:
    """
    Download all scans in a release.

    Args:
        release_scans: List of scan IDs to download.
        out_dir: Output directory.
        file_types: List of file types to download per scan.
    """
    print('Downloading 3RScan release to ' + out_dir + '...')
    for scan_id in release_scans:
        scan_out_dir = os.path.join(out_dir, scan_id)
        download_scan(scan_id, scan_out_dir, file_types)
    print('Downloaded 3RScan release.')


def download_file(url: str, out_file: str) -> None:
    """
    Download a single file from URL.

    Args:
        url: Source URL.
        out_file: Destination file path.
    """
    print(url)
    out_dir = os.path.dirname(out_file)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if not os.path.isfile(out_file):
        print('\t' + url + ' > ' + out_file)
        fh, out_file_tmp = tempfile.mkstemp(dir=out_dir)
        f = os.fdopen(fh, 'w')
        f.close()
        urllib.urlretrieve(url, out_file_tmp)
        os.rename(out_file_tmp, out_file)
    else:
        print('WARNING: skipping download of existing file ' + out_file)


def download_scan(scan_id: str, out_dir: str, file_types: List[str]) -> None:
    """
    Download a single scan with specified file types.

    Args:
        scan_id: UUID of the scan.
        out_dir: Output directory for the scan.
        file_types: List of file types to download.
    """
    print('Downloading 3RScan scan ' + scan_id + ' ...')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    for ft in file_types:
        url = DATA_URL + '/' + scan_id + '/' + ft
        out_file = out_dir + '/' + ft
        download_file(url, out_file)
    print('Downloaded scan ' + scan_id)


def download_tfrecord(url: str, out_dir: str, file: str) -> None:
    """
    Download a tfrecord file.

    Args:
        url: Base URL for tfrecords.
        out_dir: Output directory.
        file: Filename to download.
    """
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    out_file = os.path.join(out_dir, file)
    download_file(url + '/' + file, out_file)


def main() -> None:
    """Main entry point for the download CLI."""
    parser = argparse.ArgumentParser(description='Downloads 3RScan public data release.')
    parser.add_argument('-o', '--out_dir', required=True, help='directory in which to download')
    parser.add_argument('--id', help='specific scan id to download')
    parser.add_argument('--type', help='specific file type to download')
    args = parser.parse_args()

    print('By pressing any key to continue you confirm that you have agreed to the 3RScan terms of use as described at:')
    print(TOS_URL)
    print('***')
    print('Press any key to continue, or CTRL-C to exit.')

    release_scans = get_scans(BASE_URL + RELEASE)
    test_scans = get_scans(BASE_URL + HIDDEN_RELEASE)
    file_types = FILETYPES
    file_types_test = TEST_FILETYPES

    if args.type:  # download specific file type
        file_type = args.type
        if file_type == 'tfrecords':
            download_tfrecord(BASE_URL, args.out_dir, 'val-scans.tfrecords')
            download_tfrecord(BASE_URL, args.out_dir, 'train-scans.tfrecords')
            return
        elif file_type not in FILETYPES:
            print('ERROR: Invalid file type: ' + file_type)
            return
        file_types = [file_type]
        if file_type not in TEST_FILETYPES:
            file_types_test = []
        else:
            file_types_test = [file_type]

    if args.id:  # download single scan
        scan_id = args.id
        if scan_id not in release_scans and scan_id not in test_scans:
            print('ERROR: Invalid scan id: ' + scan_id)
        else:
            out_dir = os.path.join(args.out_dir, scan_id)
            if scan_id in release_scans:
                download_scan(scan_id, out_dir, file_types)
            elif scan_id in test_scans:
                download_scan(scan_id, out_dir, file_types_test)
    else:  # download entire release
        if len(file_types) == len(FILETYPES):
            print('WARNING: You are downloading the entire 3RScan release which requires ' + RELEASE_SIZE + ' of space.')
        else:
            print('WARNING: You are downloading all 3RScan scans of type ' + file_types[0])
        print('Note that existing scan directories will be skipped. Delete partially downloaded directories to re-download.')
        print('***')
        print('Press any key to continue, or CTRL-C to exit.')
        key = input('')
        download_release(release_scans, args.out_dir, file_types)
        download_release(test_scans, args.out_dir, file_types_test)


if __name__ == "__main__":
    main()
