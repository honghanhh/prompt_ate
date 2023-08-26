import re
import os
import glob
import numpy as np
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

def split_list(lst, sep):
    return [i.split() for i in ' '.join(lst).split(sep)]

def get_entities(seq):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC', 'I-PER']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3], ['PER', 4, 4]]
    """
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        tag = chunk[0]
        type_ = chunk.split('-')[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return set(chunks)


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start

def template_format_with_non_entity(path):
    df = pd.read_csv(path, 
                     header=None, 
                     delimiter = '\t', 
                     skip_blank_lines=False).rename(columns={0: 'words',1: 'raw_tags'})
    df = df.fillna('This is an end of a sentence.')
    df['tags'] = pd.Series(dtype = 'object')
    for i in range(len(df)):
        if df['raw_tags'].iloc[i] == "B" or df['raw_tags'].iloc[i] == "I": 
            df['tags'].iloc[i] = df['raw_tags'].iloc[i] + '-Term'
        else:
            df['tags'].iloc[i] = df['raw_tags'].iloc[i]
    df = pd.DataFrame({'tokens': split_list(df['words'].tolist(), 'This is an end of a sentence.'),
                       'labels': split_list(df['tags'].tolist(), 'This is an end of a sentence.')}
                     )
    df['input_text'] = [' '.join(x).strip() for x in df['tokens']]
    df = df[df['input_text'].str.len() > 0]
    df['entities'] = [get_entities(x) for x in df['labels']]
    df['target_text'] = pd.Series(dtype='object')
    df['texts_wo_terms'] = df['input_text'].copy()
    for i in range(len(df)):
        li = []
        for x in df.entities.iloc[i]:
            text = ' '.join(df.tokens.iloc[i][x[1]:x[2]+1])
            li.append(text + ' is a term')
            df['texts_wo_terms'].iloc[i] = df['texts_wo_terms'].iloc[i].replace(text, '')
        df['target_text'].iloc[i] = li
    df['target_len'] = [len(x) for x in df['target_text']]
    df = df[df['target_len'] > 0]
    df['token_wo_term'] = [[y + ' is not a term' for y in x.split()] for x, z in zip(df['texts_wo_terms'], df['entities'])]
    df['target_text'] += df['token_wo_term']
    return df[['input_text','target_text']].explode('target_text')

def concat_files(path):
    all_files = glob.glob(os.path.join(path , "*.tsv"))
    li = []
    for filename in all_files:
        df = template_format_with_non_entity(filename)
        li.append(df)
    df = pd.concat(li)
    return df

root_path = './ACTER/'
mid_path = '/annotated/annotations/sequential_annotations/iob_annotations/'

path = [root_path + 'en/corp' + mid_path + 'without_named_entities',
        root_path + 'en/wind' + mid_path + 'without_named_entities',
        root_path + 'en/equi' + mid_path + 'without_named_entities',
        root_path + 'en/htfl' + mid_path + 'without_named_entities',

        root_path + 'fr/corp' + mid_path + 'without_named_entities',
        root_path + 'fr/wind' + mid_path + 'without_named_entities',
        root_path + 'fr/equi' + mid_path + 'without_named_entities',
        root_path + 'fr/htfl' + mid_path + 'without_named_entities',

        root_path + 'nl/corp' + mid_path + 'without_named_entities',
        root_path + 'nl/wind' + mid_path + 'without_named_entities',
        root_path + 'nl/equi' + mid_path + 'without_named_entities',
        root_path + 'nl/htfl' + mid_path + 'without_named_entities',

        root_path + 'en/corp' + mid_path + 'with_named_entities',
        root_path + 'en/wind' + mid_path + 'with_named_entities',
        root_path + 'en/equi' + mid_path + 'with_named_entities',
        root_path + 'en/htfl' + mid_path + 'with_named_entities',

        root_path + 'fr/corp' + mid_path + 'with_named_entities',
        root_path + 'fr/wind' + mid_path + 'with_named_entities',
        root_path + 'fr/equi' + mid_path + 'with_named_entities',
        root_path + 'fr/htfl' + mid_path + 'with_named_entities',

        root_path + 'nl/corp' + mid_path + 'with_named_entities',
        root_path + 'nl/wind' + mid_path + 'with_named_entities',
        root_path + 'nl/equi' + mid_path + 'with_named_entities',
        root_path + 'nl/htfl' + mid_path + 'with_named_entities',
       ]

output_dir = ['en_corp_ann.csv', 'en_wind_ann.csv', 'en_equi_ann.csv', 'en_htfl_ann.csv',
              'fr_corp_ann.csv', 'fr_wind_ann.csv', 'fr_equi_ann.csv', 'fr_htfl_ann.csv',
              'nl_corp_ann.csv', 'nl_wind_ann.csv', 'nl_equi_ann.csv', 'nl_htfl_ann.csv',
              
              'en_corp_nes.csv', 'en_wind_nes.csv', 'en_equi_nes.csv', 'en_htfl_nes.csv',
              'fr_corp_nes.csv', 'fr_wind_nes.csv', 'fr_equi_nes.csv', 'fr_htfl_nes.csv',
              'nl_corp_nes.csv', 'nl_wind_nes.csv', 'nl_equi_nes.csv', 'nl_htfl_nes.csv',
             ]

for x, y in zip(path, output_dir):
    print(y.split('.')[0])
    df = concat_files(x)
    terms = [x for x in df['target_text'] if 'is a term' in x]
    non_terms = [x for x in df['target_text'] if 'is not a term' in x]
    print(len(df), len(terms), len(non_terms), round(len(terms)/len(non_terms),2))
    # df.to_csv('../data/template_datasets/' + y, index=False)
