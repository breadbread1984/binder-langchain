#!/usr/bin/python3

def create_table_prompt(df, title):
  string = "CREATE TABLE %s(\n" % title
  for idx, header in enumerate(df.columns):
    column_type = {'int64':'int',
                   'float64':'real',
                   'datetime64':'datetime',
                   'text':'text'}[df[header].dtype]
    if idx != len(df.columns) - 1:
      string += "\t%s %s,\n" % (header, column_type)
    else:
      string += "\t%s %s)\n" % (header, column_type)
  return string

def sql_example(prompt_style, df, num_rows, few_shot_demonstration = True):
  if prompt_style == 'select_full_table':
    string = '/*\nAll rows of the table:\nSELECT * FROM w;\n'
  elif prompt_style == 'select_3':
    string = '/*\n{} example rows:\nSELECT * FROM w LIMIT {};\n'.format(num_rows, num_rows)
  elif few_shot_demonstration is True and prompt_style in \
              ["select_3_full_table",
               "select_3_full_table_w_gold_passage_image",
               "select_3_full_table_w_all_passage_image"]:
    string = '/*\n{} example rows:\nSELECT * FROM w LIMIT {};\n'.format(num_rows, num_rows)
  elif few_shot_demonstration is False and prompt_style in \
              ["select_3_full_table",
               "select_3_full_table_w_gold_passage_image",
               "select_3_full_table_w_all_passage_image"]:
    string = '/*\nAll rows of the table:\nSELECT * FROM w;\n'
  else:
    raise ValueError(f"Select x prompt style {self.prompt_style} is not supported.")

  for column_id, header in enumerate(df.columns):
    string += str(header)
    if column_id != len(df.columns) - 1:
      string += '\t'
  string += '\n'

  for row_id, row in df.iloc[:num_rows].iterrows():
    for column_id, header in enumerate(df.columns):
      string += str(row[header])
      if column_id != len(df.columns) - 1:
        string += '\t'
    string += '\n'
  string += '*/\n'

  return string

def passage_prompt(passages, only_title):
    if len(passages) == 0:
      return ""
    passage_table_prompt = ""
    _header = []
    _rows = [[]]
    for passage in passages:
      _header.append(passage['title'])
      _rows[0].append(passage['text'])
    passage_table = prepare_df_for_neuraldb_from_table({"header": _header, "rows": _rows})
    passage_table_prompt += create_table_prompt(passage_table, "Passages")
    if not only_title:
      passage_table_prompt += sql_example(
		prompt_style,
        df=passage_table,
        num_rows=passage_table.shape[0]
      )
    return passage_table_prompt

def image_prompt(images, only_title):
    if len(images) == 0:
      return ""
    image_table_prompt = ""
    _header = []
    _rows = [[]]
    for image in images:
      _header.append(image['title'])
      _rows[0].append(image['caption'])
    image_table = prepare_df_for_neuraldb_from_table({"header": _header, "rows": _rows})
    image_table_prompt += create_table_prompt(image_table, "Images")
    if not only_title:
      image_table_prompt += sql_example(
		prompt_style,
        df=image_table,
        num_rows=image_table.shape[0]
      )
    return image_table_prompt
