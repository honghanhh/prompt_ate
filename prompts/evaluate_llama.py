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

def eval_template3(df, col, gold_path):
    output =  postprocess(df, col)
    output = [extract_between_markers(x) for x in output] 
    predictions = []
    for x in output:
        predictions.extend(x)

    gold_list = pd.read_csv(gold_path, header=None, delimiter='\t')[0].tolist()
    _, _, _, precision, recall, fscore = computeTermEvalMetrics(predictions, gold_list)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data_path", type=str, required=True)
    parser.add_argument("--format", default="format", type=str, required=True)
    parser.add_argument("--lang", default="language", type=str, required=True)
    parser.add_argument("--ver", default="version", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    # print(df.head(2))
    gold_list = read_data(args.lang, args.ver)

    if args.lang == 'en':
        if args.ver == 'ann':
            print("English ANN evaluation")

            ##########################################
            print("#"*50)
            print("#1. Extracted IOB format")
            precision, recall, fscore = eval_template1(df, args.format, gold_list)

            ##########################################
            print("#"*50)
            print("#2. Extracted candidate term list")
            en_ann_2 = []
            for x in df[args.format]:
                term = eval(str(x).split('Output: ')[1]) if len(str(x).split('Output: ')) > 1 else []
                en_ann_2.extend(term)

            _, _, _, precision, recall, fscore = computeTermEvalMetrics(en_ann_2, gold_list)

            ##########################################
            print("#"*50)
            print("#3. Masking terms")
            
            en_ann_3 = []
            for x in df[args.format]:
                sent = str(x).split('Output: ')[1].split('[INST]')[0].strip() if len(str(x).split('Output: ')) > 1 else ""
                en_ann_3.extend(extract_between_markers(sent))
            en_ann_3_updated = []
            for x in en_ann_3:
                en_ann_3_updated.append(x.split('@@')[-1])

            _, _, _, precision, recall, fscore = computeTermEvalMetrics(en_ann_3_updated, gold_list)

        elif args.ver == 'nes':
            print("English NES evaluation")

            ##########################################
            print("#"*50)
            print("#1. Extracted IOB format")
            precision, recall, fscore = eval_template1(df, args.format, gold_list)

            ##########################################
            print("#"*50)
            print("#2. Extracted candidate term list")
            en_nes_2 = []
            for x in df[args.format]:
                term = eval(str(x).split('Output: ')[1].split('[INST]')[0].strip()) if len(str(x).split('Output: ')) > 1 and len(str(x).split('Output: ')[1].split('[INST]')[0].strip()) > 1 else []
                en_nes_2.extend(term)

            _, _, _, precision, recall, fscore = computeTermEvalMetrics(en_nes_2, gold_list)

            ##########################################
            print("#"*50)
            print("#3. Masking terms")

            en_nes_3 = []
            for x in df[args.format]:
                sent = str(x).split('Output: ')[1].split('[INST]')[0].strip() if len(str(x).split('Output: ')) > 1 else ""
                en_nes_3.extend(extract_between_markers(sent))

            en_nes_3_updated = []
            for x in en_nes_3:
                en_nes_3_updated.append(x.split('@@')[-1])
    
            _, _, _, precision, recall, fscore = computeTermEvalMetrics(en_nes_3_updated, gold_list)
        else: 
            raise Exception("Version not supported")
    elif args.lang == 'fr':
        
        if args.ver == 'ann':
            print("French ANN evaluation")

            ##########################################
            print("#"*50)
            print("#1. Extracted IOB format")

            precision, recall, fscore =  eval_template1(df, args.format, gold_list)

            ##########################################
            print("#"*50)
            print("#2. Extracted candidate term list")
            
            fr_ann_2 = []
            for x in df[args.format]:
                term = eval(str(x).split('Output: ')[1].split('[INST]')[0].strip()) if len(str(x).split('Output: ')) > 1 and len(str(x).split('Output: ')[1].split('[INST]')[0].strip()) > 1 else []
                fr_ann_2.extend(term)
            _, _, _, precision, recall, fscore = computeTermEvalMetrics(fr_ann_2, gold_list)

            ##########################################
            print("#"*50)
            print("#3. Masking terms")
            fr_ann_3 = []
            for x in df[args.format]:
                sent = str(x).split('Output: ')[1].split('[INST]')[0].strip() if len(str(x).split('Output: ')) > 1 else ""
                fr_ann_3.extend(extract_between_markers(sent))
                
            _, _, _, precision, recall, fscore = computeTermEvalMetrics(fr_ann_3, gold_list)

        elif args.ver == 'nes':
            print("French NES evaluation")

            ##########################################
            print("#"*50)
            print("#1. Extracted IOB format")
            precision, recall, fscore =  eval_template1(df, args.format, gold_list)

            ##########################################
            print("#"*50)
            print("#2. Extracted candidate term list")
            fr_nes_2 = []
            for x in df[args.format]:
                term = eval(str(x).split('Output: ')[1].split('[INST]')[0].strip()) if len(str(x).split('Output: ')) > 1 and len(str(x).split('Output: ')[1].split('[INST]')[0].strip()) > 1 else []
                fr_nes_2.extend(term)
                
            _, _, _, precision, recall, fscore = computeTermEvalMetrics(fr_nes_2, gold_list)
            

            ##########################################
            print("#"*50)
            print("#3. Masking terms")
            fr_nes_3 = []
            for x in df[args.format]:
                sent = str(x).split('Output: ')[1].split('[INST]')[0].strip() if len(str(x).split('Output: ')) > 1 else ""
                fr_nes_3.extend(extract_between_markers(sent))

            _, _, _, precision, recall, fscore = computeTermEvalMetrics(fr_nes_3, gold_list)

        else: 
            raise Exception("Version not supported")
    elif args.lang == 'nl':
        
        if args.ver == 'ann':
            print("Dutch ANN evaluation")
            ##########################################
            print("#"*50)
            print("#1. Extracted IOB format")
            precision, recall, fscore =  eval_template1(df, args.format, gold_list)

            ##########################################
            print("#"*50)
            print("#2. Extracted candidate term list")
            nl_ann_2 = []
            for x in df[args.format]:
                term = eval(str(x).split('Output: ')[1].split('[INST]')[0].strip()) if len(str(x).split('Output: ')) > 1 and len(str(x).split('Output: ')[1].split('[INST]')[0].strip()) > 1 else []
                nl_ann_2.extend(term)
            _, _, _, precision, recall, fscore = computeTermEvalMetrics(nl_ann_2, gold_list)

            ##########################################
            print("#"*50)
            print("#3. Masking terms")
            nl_ann_3 = []
            for x in df[args.format]:
                sent = str(x).split('Output: ')[1].split('[INST]')[0].strip() if len(str(x).split('Output: ')) > 1 else ""
                nl_ann_3.extend(extract_between_markers(sent))

            nl_ann_3_updated = []
            for x in nl_ann_3:
                nl_ann_3_updated.append(x.split('@@')[-1])
            
            _, _, _, precision, recall, fscore = computeTermEvalMetrics(nl_ann_3, gold_list)
            

        elif args.ver == 'nes':
            print("Dutch NES evaluation")

            #########################################
            print("#"*50)
            print("#1. Extracted IOB format")
            precision, recall, fscore =  eval_template1(df, args.format, gold_list)

            ##########################################
            print("#"*50)
            print("#2. Extracted candidate term list")
            nl_nes_2 = []
            for x in df[args.format]:
                term = eval(str(x).split('Output: ')[1].split('[INST]')[0].strip()) if len(str(x).split('Output: ')) > 1 and len(str(x).split('Output: ')[1].split('[INST]')[0].strip()) > 1 else []
                nl_nes_2.extend(term)
                
            _, _, _, precision, recall, fscore = computeTermEvalMetrics(nl_nes_2, gold_list)

            ##########################################
            print("#"*50)
            print("#3. Masking terms")
            nl_nes_3 = []
            for x in df[args.format]:
                sent = str(x).split('Output: ')[1].split('[INST]')[0].strip() if len(str(x).split('Output: ')) > 1 else ""
                nl_nes_3.extend(extract_between_markers(sent))

            _, _, _, precision, recall, fscore = computeTermEvalMetrics(nl_nes_3, gold_list)

        else: 
            raise Exception("Version not supported")
        
    else:
        raise Exception("Language not supported")

    
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F-score: " + str(fscore))
