#!/usr/bin/env python3
"""Downloads LibriTTS, LJSpeech, and VCTK and builds train/test CSV manifests.

Follows the dataset split specification in Section 3.1 of the paper
(https://arxiv.org/pdf/2008.06006):
- LibriTTS: official train (train-clean-100, optionally train-clean-360 and
  train-other-500), test-clean, and test-other splits.
- LJSpeech: 90% random split for training, 10% for testing.
- VCTK: 90% per-speaker random split for training, 10% for testing across all
  109 speakers.
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tec import data_prep  # noqa: E402

LJSPEECH_URL = 'https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2'
LIBRITTS_BASE_URL = 'https://www.openslr.org/resources/60'
VCTK_URL = (
    'https://huggingface.co/datasets/arampacha/vctk_22kHz/'
    'resolve/main/raw_data.zip'
)


def _download_and_extract_tar(url: str, dest_dir: str, is_bz2: bool = False):
  os.makedirs(dest_dir, exist_ok=True)
  tar_flag = '-xj' if is_bz2 else '-xz'
  cmd = f'curl -sL "{url}" | tar {tar_flag} -C "{dest_dir}"'
  subprocess.run(cmd, shell=True, check=True)


def _download_and_extract_zip(url: str, dest_dir: str):
  os.makedirs(dest_dir, exist_ok=True)
  tmp_zip = os.path.join(dest_dir, '_temp_download.zip')
  try:
    subprocess.run(['curl', '-sL', url, '-o', tmp_zip], check=True)
    subprocess.run(['unzip', '-q', '-o', tmp_zip, '-d', dest_dir], check=True)
  finally:
    if os.path.exists(tmp_zip):
      os.remove(tmp_zip)


def main():
  parser = argparse.ArgumentParser(
      description='Download open-source datasets and build CSV manifests.')
  parser.add_argument(
      '--data_dir',
      type=str,
      required=True,
      help='Root directory to store datasets and CSV manifests.')
  parser.add_argument(
      '--libritts_train_splits',
      type=str,
      default='train-clean-100',
      help='Comma-separated LibriTTS training splits to include '
      '(e.g. train-clean-100,train-clean-360,train-other-500).')
  parser.add_argument(
      '--skip_download',
      action='store_true',
      help='Skip downloading archives and only build CSV manifests.')
  parser.add_argument(
      '--seed',
      type=int,
      default=42,
      help='Random seed for 90%%/10%% train/test splits.')
  args = parser.parse_args()

  ljspeech_dir = os.path.join(args.data_dir, 'ljspeech')
  vctk_dir = os.path.join(args.data_dir, 'vctk')
  libritts_dir = os.path.join(args.data_dir, 'libritts')
  manifest_dir = os.path.join(args.data_dir, 'manifests')
  os.makedirs(manifest_dir, exist_ok=True)

  train_splits = [
      s.strip() for s in args.libritts_train_splits.split(',') if s.strip()
  ]

  if not args.skip_download:
    print('Downloading and extracting LJSpeech-1.1...')
    _download_and_extract_tar(LJSPEECH_URL, ljspeech_dir, is_bz2=True)

    for split in ['test-clean', 'test-other'] + train_splits:
      url = f'{LIBRITTS_BASE_URL}/{split}.tar.gz'
      print(f'Downloading and extracting LibriTTS {split}...')
      _download_and_extract_tar(url, libritts_dir, is_bz2=False)

    print('Downloading and extracting VCTK...')
    _download_and_extract_zip(VCTK_URL, vctk_dir)

  lj_train_csv = os.path.join(manifest_dir, 'ljspeech_train.csv')
  lj_test_csv = os.path.join(manifest_dir, 'ljspeech_test.csv')
  n_lj_tr, n_lj_te = data_prep.build_ljspeech_manifests(
      ljspeech_dir, lj_train_csv, lj_test_csv, seed=args.seed)
  print(f'LJSpeech manifests: train={n_lj_tr}, test={n_lj_te}')

  vctk_train_csv = os.path.join(manifest_dir, 'vctk_train.csv')
  vctk_test_csv = os.path.join(manifest_dir, 'vctk_test.csv')
  n_vc_tr, n_vc_te = data_prep.build_vctk_manifests(
      vctk_dir, vctk_train_csv, vctk_test_csv, seed=args.seed)
  print(f'VCTK manifests: train={n_vc_tr}, test={n_vc_te}')

  libri_root = os.path.join(libritts_dir, 'LibriTTS')
  if not os.path.isdir(libri_root):
    libri_root = libritts_dir

  test_clean_csv = os.path.join(manifest_dir, 'libritts_test_clean.csv')
  n_tc = data_prep.build_libritts_manifest(
      [os.path.join(libri_root, 'test-clean')], test_clean_csv)
  print(f'LibriTTS test-clean manifest: {n_tc}')

  test_other_csv = os.path.join(manifest_dir, 'libritts_test_other.csv')
  n_to = data_prep.build_libritts_manifest(
      [os.path.join(libri_root, 'test-other')], test_other_csv)
  print(f'LibriTTS test-other manifest: {n_to}')

  train_split_dirs = [
      os.path.join(libri_root, s)
      for s in train_splits
      if os.path.isdir(os.path.join(libri_root, s))
  ]
  train_csv = os.path.join(manifest_dir, 'libritts_train.csv')
  n_tr = data_prep.build_libritts_manifest(train_split_dirs, train_csv)
  print(f'LibriTTS train manifest: {n_tr}')


if __name__ == '__main__':
  main()
