#!/usr/bin/python3

from absl import flags, app
from utils import load_data_split

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_enum('dataset', default = 'tab_fact', num_values = {'tab_fact', 'mmqa', 'wikiq'}, help = 'available dataset')
  flags.DEFINE_enum('split', default = 'test', num_values = {'train', 'validation', 'test'}, help = 'dataset split')
  flags.DEFINE_enum('model', default = 'qwen2', num_values = {'llama3', 'codellama', 'qwen2', 'codeqwen'}, help = 'model name')

def main(unused_argv):
  samples = load_data_split(FLAGS.dataset, split = FLAGS.split)
