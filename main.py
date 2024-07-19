#!/usr/bin/python3

from absl import flags, app
from nsql.database import NeuralDB
from utils import load_data_split
from models import *
from prompts import get_binder_template

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_enum('dataset', default = 'tab_fact', num_values = {'tab_fact', 'mmqa', 'wikiq'}, help = 'available dataset')
  flags.DEFINE_enum('split', default = 'test', num_values = {'train', 'validation', 'test'}, help = 'dataset split')
  flags.DEFINE_enum('model', default = 'qwen2', num_values = {'llama3', 'codellama', 'qwen2', 'codeqwen'}, help = 'model name')
  flags.DEFINE_enum('prompt_style', default = 'select_3_full_table', num_values = {'select_3_full_table',
                                                               'select_full_table',
                                                               'select_3',
                                                               'no_select',
                                                               'select_3_full_table_w_all_passage_image',
                                                               'select_3_full_table_w_gold_passage_image',
                                                               'no_table'}, help = 'prompt style')
  flags.DEFINE_enum('generate_type', default = 'nsql', enum_values = {'answer', 'nsql', 'sql', 'npython', 'python'}, help = 'generate type')
  flags.DEFINE_integer('n_shots', default = 8, help = 'few shot example number')
  flags.DEFINE_boolean('locally', default = False, help = 'run locally')

def main(unused_argv):
  tokenizer, llm = {
    'llama3': Llama3,
    'codellama': CodeLlama,
    'qwen2': Qwen2,
    'codeqwen': CodeQwen1_5,
  }[FLAGS.model](FLAGS.locally)
  samples = load_data_split(FLAGS.dataset, split = FLAGS.split)
  for sample in samples:
    db = NueralDB(tables = [{'title': sample['table']['page_title'], 'table': sample['table']}])
    sample['table'] = db.get_table_df()
    sample['title'] = db.get_table_title()
    template = get_binder_template(
      FLAGS.dataset,
      tokenizer,
      FLAGS.n_shots,
      prompt_style = FLAGS.prompt_style,
      generate_type = FLAGS.generate_type,
      title = sample['title'],
      table = sample['table'],
    )
    chain = template | llm
    response = chain.invoke({'question': ''})
