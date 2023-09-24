import re
import pandas as pd
import argparse

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def read_data(lang, ver):
    if ver == 'ann':
        df = pd.read_csv('../ACTER/'+ lang + '/htfl/annotated/annotations/unique_annotation_lists/htfl_'+ lang + '_terms.tsv',
                          header=None, delimiter='\t')[0].tolist()
    elif ver == 'nes':
        df = pd.read_csv('../ACTER/'+ lang + '/htfl/annotated/annotations/unique_annotation_lists/htfl_'+ lang + '_terms_nes.tsv',
                          header=None, delimiter='\t')[0].tolist()
    else:
        raise Exception("Version not supported")
    return df

def postprocess(df, col):
    res = [re.sub('\n','', str(x)) for x in df[col]]
    res =  [x.split('                ')[0] for x in res]
    res =  [x.split('Note')[0] for x in res]
    res =  ['' if x[1:4] == '"""' else x for x in res]
    return res

def computeTermEvalMetrics(extracted_terms, gold_df):
    # make lower case cause gold standard is lower case
    extracted_terms = [item.lower() for item in extracted_terms]
    extracted_terms = set([x for x in extracted_terms if x != ''])
    gold_set = [item.lower() for item in gold_df]
    gold_set =  set([x for x in gold_set if x != ''])
    true_pos = extracted_terms.intersection(gold_set)
    recall = round(len(true_pos)*100/len(gold_set),1)
    precision = round(len(true_pos)*100/len(extracted_terms),1)
    fscore = round(2*(precision*recall)/(precision+recall),1)
    print(str(precision)+' & ' +  str(recall)+' & ' +  str(fscore))
    return len(extracted_terms), len(gold_set), len(true_pos), precision, recall, fscore

### TEMPLATE 1
def filter_text(text):
    allowed_chars = {'B', 'I', 'O', ' '}
    filtered_text = ''.join(c for c in text if c in allowed_chars)
    return filtered_text

def extract_entities(words, labels):
    entities = []
    current_entity = ''
    current_label = ''

    for word, label in zip(words, labels):
        if label.startswith('B'):
            if current_entity:
                entities.append(current_entity.strip())
            current_entity = word
            current_label = label[2:]
        elif label.startswith('I') and current_label == label[2:]:
            current_entity += ' ' + word
        else:
            if current_entity:
                entities.append(current_entity.strip())
                current_entity = ''
                current_label = ''

    if current_entity:
        entities.append(current_entity.strip())

    return entities

def eval_template1(df, col, gold_list):
    output = postprocess(df, col)
    output = [re.sub("'", "", x).strip() for x in output]
    
    extracted_list = []
    for x, y in list(zip(df.words, output)):
        words = eval(x)
        labels = y[:len(words)*2]
        
        if labels.startswith("'I") or labels == '':
            labels = 'O '*len(words)
        labels =  filter_text(labels).split(' ')
        extracted_list.append(labels)

    predictions = []
    for x, y in zip(df.words, extracted_list):
        terms =  extract_entities(eval(x), y)
        predictions.extend(terms)
    _, _, _, precision, recall, fscore = computeTermEvalMetrics(predictions, gold_list)
    return precision, recall, fscore
    
### TEMPLATE 3
def extract_between_markers(input_string):
    pattern = r'@@(.*?)##'
    extracted = re.findall(pattern, input_string)
    return extracted


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data_path", type=str, required=True)
    parser.add_argument("--format", default="format", type=int, required=True)
    parser.add_argument("--lang", default="language", type=str, required=True)
    parser.add_argument("--ver", default="version", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    # print(df.head(2))
    gold_list = read_data(args.lang, args.ver)

    print(args.lang)
    print(args.ver)

    if args.format == 1:
        print("#"*50)
        print("#1. Extracted IOB format")
        precision, recall, fscore = eval_template1(df, args.format, gold_list)

    elif args.format == 2:
        print("#"*50)
        print("#2. Extracted candidate term list")
        candidate_terms = []
        if args.lang == 'en' and args.ver == 'ann':
            for x in df[args.format]:
                term = eval(str(x).split('Output: ')[1]) if len(str(x).split('Output: ')) > 1 else []
                candidate_terms.extend(term)
        elif args.lang in ['en', 'fr', 'nl'] and args.ver in ['ann', 'nes']:
            for x in df[args.format]:
                term = eval(str(x).split('Output: ')[1].split('[INST]')[0].strip()) if len(str(x).split('Output: ')) > 1 and len(str(x).split('Output: ')[1].split('[INST]')[0].strip()) > 1 else []
                candidate_terms.extend(term)
        else:
            raise Exception("Language not supported")
        _, _, _, precision, recall, fscore = computeTermEvalMetrics(candidate_terms, gold_list)
    elif args.format == 3:
        ### RAW
        candidate_terms = []
        for x in df[args.format]:
            sent = str(x).split('Output: ')[1].split('[INST]')[0].strip() if len(str(x).split('Output: ')) > 1 else ""
            candidate_terms.extend(extract_between_markers(sent))

        ### PROCESSED
        candidate_terms_list = []
        for x in candidate_terms:
            candidate_terms_list.append(x.split('@@')[-1])

        _, _, _, precision, recall, fscore = computeTermEvalMetrics(candidate_terms_list, gold_list)
        _, _, _, precision1, recall1, fscore1 = computeTermEvalMetrics(candidate_terms, gold_list)
                
        print("Precision: " + str(precision1))
        print("Recall: " + str(recall1))
        print("F-score: " + str(fscore1))

    else:
        raise Exception("Format not supported")
                
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F-score: " + str(fscore))
