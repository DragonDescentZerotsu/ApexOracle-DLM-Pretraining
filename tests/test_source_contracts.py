import hashlib
import subprocess
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
LEGACY_COMMIT = 'fda167cf5fb90ac57952482fb5c0e605b188c105'
LEGACY_TAG = 'legacy-code-snapshot-2026-08-10'
TOKENIZER_REVISION = '55e83392264cb998f7aa5014847df29868aefeb8'


def test_descriptor_statistics_are_original_and_well_formed():
  expected = {
    'descriptors_mean.npy': '19d2693e9797f848146dc27536c1861cd331008dc9fe9e003b168ad6210b5e93',
    'descriptors_std.npy': 'c69b149f0f561b4ee7669b0e3cef82fbaea2fa630142a8571fa31f0e8b6bf608',
  }
  for filename, digest in expected.items():
    path = ROOT / filename
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    values = np.load(path)
    assert values.shape == (209,)
    assert np.isfinite(values).all()
  assert (np.load(ROOT / 'descriptors_std.npy') > 0).all()


def test_selfies_config_is_explicit_and_tokenizer_is_fixed():
  config = yaml.safe_load((ROOT / 'configs/data/SELFIES.yaml').read_text())
  assert config['tokenizer_revision'] == TOKENIZER_REVISION
  assert 'APEXORACLE_DLM_TRAIN_PATH' in config['train_path']
  assert 'APEXORACLE_DLM_VALID_PATH' in config['valid_path']
  assert 'APEXORACLE_DLM_CACHE_DIR' in config['cache_dir']


def test_canonical_source_has_no_author_machine_paths():
  paths = [
    ROOT / 'dataloader.py',
    ROOT / 'diffusion.py',
    ROOT / 'models/dit.py',
  ]
  paths.extend((ROOT / 'configs/data').glob('*.yaml'))
  paths.extend((ROOT / 'scripts').glob('*.sh'))
  forbidden = ('/data1/fangping', '/data1/fangping2', '/share/kuleshov')
  for path in paths:
    text = path.read_text()
    assert not any(value in text for value in forbidden), path


def test_medium_profile_matches_unchanged_regression_head():
  model = yaml.safe_load((ROOT / 'configs/model/medium.yaml').read_text())
  source = (ROOT / 'models/dit.py').read_text()
  readme = (ROOT / 'README.md').read_text()
  assert model['hidden_size'] == 1024
  assert 'RobertaRegressionHead(1024, 0.1, 209)' in source
  assert 'model=medium' in readme


def test_legacy_tag_and_manifest_when_git_metadata_is_available():
  if not (ROOT / '.git').exists():
    return
  resolved = subprocess.run(
    ['git', 'rev-parse', f'{LEGACY_TAG}^{{commit}}'],
    cwd=ROOT,
    check=True,
    text=True,
    capture_output=True,
  ).stdout.strip()
  assert resolved == LEGACY_COMMIT
  manifest = ROOT / 'manifests/legacy_source.sha256'
  for line in manifest.read_text().splitlines():
    if not line or line.startswith('#'):
      continue
    digest, relative = line.split('  ', 1)
    content = subprocess.run(
      ['git', 'show', f'{LEGACY_TAG}:{relative}'],
      cwd=ROOT,
      check=True,
      capture_output=True,
    ).stdout
    assert hashlib.sha256(content).hexdigest() == digest
