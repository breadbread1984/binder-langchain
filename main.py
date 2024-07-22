#!/usr/bin/python3

from absl import flags, app
from tqdm import tqdm
from nsql.database import NeuralDB
from nsql.parser import extract_answers
from utils import load_data_split, Evaluator
from models import *
from prompts import get_binder_template

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_enum('dataset', default = 'tab_fact', enum_values = {'tab_fact', 'mmqa', 'wikiq'}, help = 'available dataset')
  flags.DEFINE_enum('split', default = 'test', enum_values = {'train', 'validation', 'test'}, help = 'dataset split')
  flags.DEFINE_enum('model', default = 'llama3', enum_values = {'llama3', 'codellama', 'qwen2', 'codeqwen'}, help = 'model name')
  flags.DEFINE_enum('prompt_style', default = 'select_3_full_table', enum_values = {'select_3_full_table',
                                                               'select_full_table',
                                                               'select_3',
                                                               'no_select',
                                                               'select_3_full_table_w_all_passage_image',
                                                               'select_3_full_table_w_gold_passage_image',
                                                               'no_table'}, help = 'prompt style')
  flags.DEFINE_enum('generate_type', default = 'nsql', enum_values = {'answer', 'nsql', 'sql', 'npython', 'python'}, help = 'generate type')
  flags.DEFINE_integer('n_shots', default = 8, help = 'few shot example number')
  flags.DEFINE_boolean('locally', default = False, help = 'run locally')

def parser(response):
  if response.startswith('assistant\n\n'):
    response = response.replace('assistant\n\n','')
  return response

def main(unused_argv):
  tokenizer, llm = {
    'llama3': Llama3,
    'codellama': CodeLlama,
    'qwen2': Qwen2,
    'codeqwen': CodeQwen1_5,
  }[FLAGS.model](FLAGS.locally)
  samples = load_data_split(FLAGS.dataset, split = FLAGS.split)
  score = 0
  for sample in tqdm(samples):
    db = NeuralDB(tables = [{'title': sample['table']['page_title'], 'table': sample['table']}])
    template = get_binder_template(
      FLAGS.dataset,
      tokenizer,
      FLAGS.n_shots,
      prompt_style = FLAGS.prompt_style,
      generate_type = FLAGS.generate_type,
      title = db.get_table_title(),
      table = db.get_table_df(),
    )
    chain = template | llm | parser
    sql = chain.invoke({'question': sample['question']})
    try:
      sub_table = db.execute_query(sql)
    except e as Exception:
      print(e)
      continue
    answer = extract_answers(sub_table)
    if isinstance(answer, str): answer = [answer]
    score += Evaluator().evaluate(answer, sample['label'], dataset = FLAGS.dataset, question = sample['question'])
  print('score: %f' % (score / len(samples)))

if __name__ == "__main__":
  add_options()
  app.run(main)
