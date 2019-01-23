from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.scut import scut
import numpy as np

# Set up scut_<part>_<split>
for part in ['Part_A', 'Part_B']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = '{}_{}'.format(part, split)
    __sets[name] = (lambda split=split, part=part: scut(split, part))


def get_imdb(name):
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  return list(__sets.keys())
