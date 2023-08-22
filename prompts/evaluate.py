import re
import pandas as pd
import argparse

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
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
    # print(str(precision)+' & ' +  str(recall)+' & ' +  str(fscore))
    return len(extracted_terms), len(gold_set), len(true_pos), precision, recall, fscore

def extract_between_markers(input_string):
    pattern = r'@@(.*?)##'
    extracted = re.findall(pattern, input_string)
    return extracted

def read_data(lang, ver):
    if ver == 'ann':
        df = pd.read_csv('./ACTER/'+ lang + '/htfl/annotated/annotations/unique_annotation_lists/htfl_'+ lang + '_terms.tsv',
                          header=None, delimiter='\t')[0].tolist()
    elif ver == 'nes':
        df = pd.read_csv('./ACTER/'+ lang + '/htfl/annotated/annotations/unique_annotation_lists/htfl_'+ lang + '_terms_nes.tsv',
                          header=None, delimiter='\t')[0].tolist()
    else:
        raise Exception("Version not supported")
    return df

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

def filter_text(text):
    allowed_chars = {'B', 'I', 'O', ' '}
    filtered_text = ''.join(c for c in text if c in allowed_chars)
    return filtered_text

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data_path", type=str, required=True)
    parser.add_argument("--lang", default="language", type=str, required=True)
    parser.add_argument("--ver", default="version", type=str, required=True)
    # parser.add_argument("--format", default="format", type=str, required=True)
    parser.add_argument("--output_path", default="output_path", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    gold_list = read_data(args.lang, args.ver)

    if args.lang == 'en':
        if args.ver == 'ann':
            print("English ANN evaluation")
            ##########################################
            print("#1. Extracted candidate term list")
            en_ann_output1 = []
            for x in df['en_ann_output2']:
                if x.startswith('[') and x.endswith(']') and x.count('[') == 1:
                    # print(x)
                    if "Jehovah's Witness" in x:
                        en_ann_output1.extend(["18-year-old", "Jehovah's Witness", "sickle cell disease", "life-threatening anemia"])
                    else:
                        en_ann_output1.extend(eval(x))
            _, _, _, precision, recall, fscore = computeTermEvalMetrics(en_ann_output1, gold_list)
            print("Precision: " + str(precision))
            print("Recall: " + str(recall))
            print("F-score: " + str(fscore))

            ##########################################
            print("#2. Extracted candidate term list")
            en_ann_output2 = []
            for x, y in list(zip(df.words, df.en_ann_output2)):
                words = eval(x)
                labels = y[:len(words)*2]
                if labels.startswith("'I"):
                    labels = 'O '*len(words)
                labels =  filter_text(labels).split(' ')
                en_ann_output2.append(labels)

            en_ann_output2_processed = []
            for x, y in zip(df.words, df.en_ann_output2_processed):
                terms =  extract_entities(eval(x), y)
                en_ann_output2_processed.extend(terms)

            _, _, _, precision, recall, fscore = computeTermEvalMetrics(en_ann_output2, gold_list)
            print("Precision: " + str(precision))
            print("Recall: " + str(recall))
            print("F-score: " + str(fscore))

            ##########################################
            print("#3. Masking terms")
            df['en_ann_output3_processed'] = [extract_between_markers(x) for x in df['en_ann_output3']]
            en_ann_output3_processed = []
            for x in df['en_ann_output1_processed']:
                en_ann_output3_processed.extend(x)
            
            _, _, _, precision, recall, fscore = computeTermEvalMetrics(en_ann_output3_processed, gold_list)
            print("Precision: " + str(precision))
            print("Recall: " + str(recall))
            print("F-score: " + str(fscore))
        elif args.ver == 'nes':
            print("English NES evaluation")
            ##########################################
            print("#1. Extracted candidate term list")
            en_nes_output1 = []
            for x in df['en_nes_output2']:
                if x.startswith('[') and x.endswith(']') and x.count('[') == 1:
                    # print(x)
                    if "E/e'septal" in x:
                        en_nes_output1.extend(["E/e'septal", "PCWP", "E/e'lateral", "E/e'mean"])
                    elif "['cutoff', 'predictive value', 'E/e' septal', 'negative likelihood ratio']" in x:
                        en_nes_output1.extend(["cutoff", "predictive value", "E/e' septal", "negative likelihood ratio"])
                    else:
                        en_nes_output1.extend(eval(x))
        
            _, _, _, precision, recall, fscore = computeTermEvalMetrics(en_nes_output1, gold_list)
            print("Precision: " + str(precision))
            print("Recall: " + str(recall))
            print("F-score: " + str(fscore))
            ##########################################
            print("#2. Extracted candidate term list")
            en_nes_output2 = []
            for x, y in list(zip(df.words, df.en_nes_output2)):
                words = eval(x)
                labels = y[:len(words)*2]
                if labels.startswith("'I"):
                    labels = 'O '*len(words)
                labels =  filter_text(labels).split(' ')
                en_nes_output2.append(labels)

            en_nes_output2_processed = []
            for x, y in zip(df.words, df.en_nes_output2_processed):
                terms =  extract_entities(eval(x), y)
                en_nes_output2_processed.extend(terms)
                
            _, _, _, precision, recall, fscore = computeTermEvalMetrics(en_nes_output2, gold_list)
            print("Precision: " + str(precision))
            print("Recall: " + str(recall))
            print("F-score: " + str(fscore))
            ##########################################
            print("#3. Masking terms")
            
            df['en_nes_output1_processed'] = [extract_between_markers(x) for x in df['en_nes_output1']]
            en_nes_output3_processed = []
            for x in df['en_nes_output1_processed']:
                en_nes_output3_processed.extend(x)

            _, _, _, precision, recall, fscore = computeTermEvalMetrics(en_nes_output3_processed, gold_list)
            print("Precision: " + str(precision))
            print("Recall: " + str(recall))
            print("F-score: " + str(fscore))
        else: 
            raise Exception("Version not supported")
    elif args.lang == 'fr':
        ### PASS
    elif args.lang == 'nl':
        ### PASS
    else:
        raise Exception("Language not supported")

    
