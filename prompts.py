#!/usr/bin/python3

import json
import pandas as pd
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts.promtp import PromptTemplate
from langchain.output_parsers.regex import RegexParser
from utils import *

def get_binder_template(dataset,
                        tokenizer,
                        n_shots = 8,
                        prompt_style = 'select_3_full_table',
                        generate_type = 'answer',
                        title: str = None,
                        table: pd.DataFrame = None,
                        passages: dict = None,
                        images: dict = None,
                        supporting_context: dict = None,
                        only_title = False):
  assert dataset in {'mmqa', 'tab_fact', 'wikiq'}
  assert prompt_style in {'select_3_full_table',
                          'select_full_table',
                          'select_3',
                          'no_select',
                          'select_3_full_table_w_all_passage_image',
                          'select_3_full_table_w_gold_passage_image',
                          'no_table'}
  assert generate_type in {'answer', 'nsql', 'sql', 'npython', 'python'}
  system_message = "I will give you some x-y examples followed by a x, you need to give me the y, and no other content."
  # few shot cases
  user_message = few_shot_case(dataset, n_shots) + "\n\n"
  # instruction
  if generate_type == 'answer':
    user_message += """\n-- Answer the question based on the given table below.\n\n"""
  elif generate_type == 'nsql':
    user_message += """\n-- Parse the question into NeuralSQL based on the given table below.\n\n"""
  elif generate_type == 'sql':
    user_message += """\n-- Parse the question into SQL based on the given table below.\n\n"""
  elif generate_type == 'npython':
    user_message += """\n-- Parse the question into NeuralPython based on the given table below.\n\n"""
  elif generate_type == 'python':
    user_message += """\n-- Parse the question into Python based on the given table below.\n\n"""
  else:
    raise NotImplementedError

  if prompt_style != 'no_table':
    # table structure described by sql
    user_message += create_table_prompt(df, title)
  # sql example and its execution result
  if prompt_style in ['select_full_table', 'select_3_full_table']:
    user_message += sql_example(prompt_style, df = table, num_rows = table.shape[0], few_shot_demonstration = False)
  elif prompt_style in ['select_3']:
    user_message += sql_example(prompt_style, df = table, num_rows = 3, few_shot_demonstration = False)
  elif prompt_style in ['no_select', 'no_table']:
    pass
  elif prompt_style in ['select_3_full_table_w_all_passage_image','select_3_full_table_w_gold_passage_image']:
    assert dataset == 'mmqa'
    assert passages is not None and images is not None
    if prompt_style == 'select_3_full_table_w_gold_passage_image': assert supporting_context is not None
    user_message += sql_example(prompt_style, df = table, num_rows = table.shape[0], few_shot_deomonstration = False)
    all_passages, all_images = list(), list()
    with open(join('datasets','mmqa_captions.json'),'r') as f:
      caption_map = json.load(f)
    if prompt_style == 'select_3_full_table_w_all_passage_image':
      for passage_idx in range(len(passages['id'])):
        all_passages.append({
          'id': passages['id'][passage_idx],
          'title': passages['title'][passage_idx],
          'url': passages['url'][passage_idx],
          'text': passages['text'][passage_idx]
        })

      for image_idx in range(len(images['id'])):
        all_images.append({
          "id": images['id'][image_idx],
          "title": images['title'][image_idx],
          "url": images['url'][image_idx],
          "path": images['path'][image_idx],
          "pic": images['pic'][image_idx],
          "caption": caption_map[images['id'][image_idx]]
        })
    else:
      for doc_id, doc_part in zip(supporting_context['doc_id'], supporting_context['doc_part']):
        if doc_part == 'text':
          passage_idx = passages['id'].index(doc_id)
          all_passages.append({
            'id': passages['id'][passage_idx],
            'title': passages['title'][passage_idx],
            'url': passages['url'][passage_idx],
            'text': passages['text'][passage_idx]
          })
        elif doc_part == 'image':
          image_idx = images['id'].index(doc_id)
          all_images.append({
            "id": images['id'][image_idx],
            "title": images['title'][image_idx],
            "url": images['url'][image_idx],
            "path": images['path'][image_idx],
            "pic": images['pic'][image_idx],
            "caption": caption_map[doc_id]
          })
	user_message += passage_prompt(passages = all_passages, only_title = only_title)
	user_message += image_prompt(images = all_images, only_title = only_title)
  else:
	raise NotImplementedError
  system_message = system_message.repalce('{','{{')
  system_message = system_message.replace('}','}}')
  user_message = user_message.replace('{','{{')
  user_message = user_message.replace('}','}}')
  user_message += "Q: {question}\n" + \
	{
	  'answer': 'A: ',
	  'nsql': 'NeuralSQL: ',
	  'sql': 'SQL: ',
	  'npython': 'NeuralPython: ',
	  'python': 'Python: '
	}[generate_type]
  messages = [
    {'role': 'system', 'content': system_message},
    {'role': 'user', 'content': user_message}
  ]
  prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generating_prompt = True)
  template = PromptTemplate(template = prompt, input_variables = ['question'])
  return template

def few_shot_case(dataset = "tab_fact", n_shots = 8):
  file_path = {
	'mmqa': join('datasets', 'mmqa_binder.txt'),
	'tab_fact': join('datasets', 'tab_fact_binder.txt'),
	'wikiq': join('datasets', 'wikiq_binder.txt')
  }[dataset]
  with open(file_path, 'r') as f:
	lines = f.readlines()
  few_shot_prompt_list = []
  one_shot_prompt = ''
  last_line = None
  for line in lines:
	if line == '\n' and last_line == '\n':
      few_shot_prompt_list.append(one_shot_prompt)
      one_shot_prompt = ''
    else:
      one_shot_prompt += line
    last_line = line
  few_shot_prompt_list.append(one_shot_prompt)
  few_shot_prompt_list = few_shot_prompt_list[:n_shots]
  few_shot_prompt_list[-1] = few_shot_prompt_list[
    -1].strip()  # It is essential for prompting to remove extra '\n'
  few_shot_prompt = '\n'.join(few_shot_prompt_list)
  return few_shot_prompt

if __name__ == "__main__":
  pass
