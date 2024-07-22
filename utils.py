#!/usr/bin/python3

from os.path import join
from typing import Dict
import pandas as pd
from datasets import load_dataset
from nsql.normalizer import str_normalize
from datasets.wtq_evaluator import to_value_list, check_denotation
from datasets.mmqa_evaluator import acc

def load_data_split(dataset_to_load, split):
  dataset_split_loaded = load_dataset(path = join('datasets', '%s.py' % dataset_to_load), cache_dir = join('datasets', "data"), trust_remote_code = True)[split]
  samples = list()
  if dataset_to_load == 'tab_fact':
    for data_item in dataset_split_loaded:
      data_item['question'] = data_item['statement']
      data_item['answer_text'] = data_item['label']
      data_item['table']['page_title'] = data_item['table']['caption']
      samples.append(data_item)
  elif dataset_to_load == 'mmqa':
    for data_item in dataset_split_loaded:
      data_item['table']['page_title'] = data_item['table']['title']
      samples.append(data_item)
  else:
    raise NotImplementedError
  return samples

def create_table_prompt(df, title):
  string = "CREATE TABLE %s(\n" % title
  for idx, header in enumerate(df.columns):
    column_type = {'int64':'int',
                   'float64':'real',
                   'datetime64':'datetime',
                   'object':'text'}[str(df[header].dtype)]
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

def prepare_df_for_neuraldb_from_table(table: Dict, add_row_id=True, normalize=True, lower_case=True):
    header, rows = table['header'], table['rows']
    if add_row_id and 'row_id' not in header:
        header = ["row_id"] + header
        rows = [["{}".format(i)] + row for i, row in enumerate(rows)]
    if normalize:
        df = convert_df_type(pd.DataFrame(data=rows, columns=header), lower_case=lower_case)
    else:
        df = pd.DataFrame(data=rows, columns=header)

    return df

def convert_df_type(df: pd.DataFrame, lower_case=True):
    """
    A simple converter of dataframe data type from string to int/float/datetime.
    """

    def get_table_content_in_column(table):
        if isinstance(table, pd.DataFrame):
            header = table.columns.tolist()
            rows = table.values.tolist()
        else:
            # Standard table dict format
            header, rows = table['header'], table['rows']
        all_col_values = []
        for i in range(len(header)):
            one_col_values = []
            for _row in rows:
                one_col_values.append(_row[i])
            all_col_values.append(one_col_values)
        return all_col_values

    # Rename empty columns
    new_columns = []
    for idx, header in enumerate(df.columns):
        if header == '':
            new_columns.append('FilledColumnName')  # Fixme: give it a better name when all finished!
        else:
            new_columns.append(header)
    df.columns = new_columns

    # Rename duplicate columns
    new_columns = []
    for idx, header in enumerate(df.columns):
        if header in new_columns:
            new_header, suffix = header, 2
            while new_header in new_columns:
                new_header = header + '_' + str(suffix)
                suffix += 1
            new_columns.append(new_header)
        else:
            new_columns.append(header)
    df.columns = new_columns

    # Recognize null values like "-"
    null_tokens = ['', '-', '/']
    for header in df.columns:
        df[header] = df[header].map(lambda x: str(None) if x in null_tokens else x)

    # Convert the null values in digit column to "NaN"
    all_col_values = get_table_content_in_column(df)
    for col_i, one_col_values in enumerate(all_col_values):
        all_number_flag = True
        for row_i, cell_value in enumerate(one_col_values):
            try:
                float(cell_value)
            except Exception as e:
                if not cell_value in [str(None), str(None).lower()]:
                    # None or none
                    all_number_flag = False
        if all_number_flag:
            _header = df.columns[col_i]
            df[_header] = df[_header].map(lambda x: "NaN" if x in [str(None), str(None).lower()] else x)

    # Normalize cell values.
    for header in df.columns:
        df[header] = df[header].map(lambda x: str_normalize(x))

    # Strip the mis-added "01-01 00:00:00"
    all_col_values = get_table_content_in_column(df)
    for col_i, one_col_values in enumerate(all_col_values):
        all_with_00_00_00 = True
        all_with_01_00_00_00 = True
        all_with_01_01_00_00_00 = True
        for row_i, cell_value in enumerate(one_col_values):
            if not str(cell_value).endswith(" 00:00:00"):
                all_with_00_00_00 = False
            if not str(cell_value).endswith("-01 00:00:00"):
                all_with_01_00_00_00 = False
            if not str(cell_value).endswith("-01-01 00:00:00"):
                all_with_01_01_00_00_00 = False
        if all_with_01_01_00_00_00:
            _header = df.columns[col_i]
            df[_header] = df[_header].map(lambda x: x[:-len("-01-01 00:00:00")])
            continue

        if all_with_01_00_00_00:
            _header = df.columns[col_i]
            df[_header] = df[_header].map(lambda x: x[:-len("-01 00:00:00")])
            continue

        if all_with_00_00_00:
            _header = df.columns[col_i]
            df[_header] = df[_header].map(lambda x: x[:-len(" 00:00:00")])
            continue

    # Do header and cell value lower case
    if lower_case:
        new_columns = []
        for header in df.columns:
            lower_header = str(header).lower()
            if lower_header in new_columns:
                new_header, suffix = lower_header, 2
                while new_header in new_columns:
                    new_header = lower_header + '-' + str(suffix)
                    suffix += 1
                new_columns.append(new_header)
            else:
                new_columns.append(lower_header)
        df.columns = new_columns
        for header in df.columns:
            # df[header] = df[header].map(lambda x: str(x).lower())
            df[header] = df[header].map(lambda x: str(x).lower().strip())

    # Recognize header type
    for header in df.columns:

        float_able = False
        int_able = False
        datetime_able = False

        # Recognize int & float type
        try:
            df[header].astype("float")
            float_able = True
        except:
            pass

        if float_able:
            try:
                if all(df[header].astype("float") == df[header].astype(int)):
                    int_able = True
            except:
                pass

        if float_able:
            if int_able:
                df[header] = df[header].astype(int)
            else:
                df[header] = df[header].astype(float)

        # Recognize datetime type
        try:
            df[header].astype("datetime64")
            datetime_able = True
        except:
            pass

        if datetime_able:
            df[header] = df[header].astype("datetime64")

    return df

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

class Evaluator:
    def __init__(self):
        pass

    def evaluate(
            self,
            pred_answer,
            gold_answer,
            dataset,
            allow_semantic=True,
            question=None
    ):
        if dataset == 'wikitq':
            return self.eval_ex_match(pred_answer, gold_answer, allow_semantic, question)
        elif dataset == 'tab_fact':
            return self.eval_tabfact_match(pred_answer, gold_answer)
        elif dataset == 'mmqa':
            # For more metrics on MMQA,
            # please use the utils/mmqa/eval_mmqa.py to call official on all prediction data
            return self.eval_mmqa_match(pred_answer, gold_answer)
        else:
            raise ValueError(f'{dataset} evaluator is not supported.')

    def eval_ex_match(self, pred, gold, allow_semantic=True, question=None):
        if not isinstance(pred, list):
            pred = [pred]
            gold = [gold]

        pred = [str(p).lower().strip() for p in pred]
        gold = [str(g).lower().strip() for g in gold]

        if not allow_semantic:
            # WikiTQ eval w. string normalization using recognizer
            pred = [str_normalize(span) for span in pred]
            gold = [str_normalize(span) for span in gold]
            pred = to_value_list(pred)
            gold = to_value_list(gold)
            return check_denotation(pred, gold)
        else:
            assert isinstance(question, str)
            question = re.sub('\s+', ' ', question).strip().lower()
            pred = [str_normalize(span) for span in pred]
            gold = [str_normalize(span) for span in gold]
            pred = sorted(list(set(pred)))
            gold = sorted(list(set(gold)))
            # (1) 0 matches 'no', 1 matches 'yes'; 0 matches 'more', 1 matches 'less', etc.
            if len(pred) == 1 and len(gold) == 1:
                if (pred[0] == '0' and gold[0] == 'no') \
                        or (pred[0] == '1' and gold[0] == 'yes'):
                    return True
                question_tokens = question.split()
                try:
                    pos_or = question_tokens.index('or')
                    token_before_or, token_after_or = question_tokens[pos_or - 1], question_tokens[pos_or + 1]
                    if (pred[0] == '0' and gold[0] == token_after_or) \
                            or (pred[0] == '1' and gold[0] == token_before_or):
                        return True
                except Exception as e:
                    pass
            # (2) Number value (allow units) and Date substring match
            if len(pred) == 1 and len(gold) == 1:
                NUMBER_UNITS_PATTERN = re.compile('^\$*[+-]?([0-9]*[.])?[0-9]+(\s*%*|\s+\w+)$')
                DATE_PATTERN = re.compile('[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}\s*([0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2})?')
                DURATION_PATTERN = re.compile('(P|PT)(\d+)(Y|M|D|H|S)')
                p, g = pred[0], gold[0]
                # Restore `duration` type, e.g., from 'P3Y' -> '3'
                if re.match(DURATION_PATTERN, p):
                    p = re.match(DURATION_PATTERN, p).group(2)
                if re.match(DURATION_PATTERN, g):
                    g = re.match(DURATION_PATTERN, g).group(2)
                match = False
                num_flag, date_flag = False, False
                # Number w. unit match after string normalization.
                # Either pred or gold being number w. units suffices it.
                if re.match(NUMBER_UNITS_PATTERN, p) or re.match(NUMBER_UNITS_PATTERN, g):
                    num_flag = True
                # Date match after string normalization.
                # Either pred or gold being date suffices it.
                if re.match(DATE_PATTERN, p) or re.match(DATE_PATTERN, g):
                    date_flag = True
                if num_flag:
                    p_set, g_set = set(p.split()), set(g.split())
                    if p_set.issubset(g_set) or g_set.issubset(p_set):
                        match = True
                if date_flag:
                    p_set, g_set = set(p.replace('-', ' ').split()), set(g.replace('-', ' ').split())
                    if p_set.issubset(g_set) or g_set.issubset(p_set):
                        match = True
                if match:
                    return True
            pred = to_value_list(pred)
            gold = to_value_list(gold)
            return check_denotation(pred, gold)

    def eval_tabfact_match(self, pred, gold):
        if isinstance(pred, list):
            pred = pred[0]
        pred, gold = str(pred), str(gold)
        return pred == gold

    def eval_mmqa_match(self, pred_answer, gold_answer):
        return acc(pred_answer, gold_answer)
