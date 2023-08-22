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
        df = pd.read_csv('../ACTER/'+ lang + '/htfl/annotated/annotations/unique_annotation_lists/htfl_'+ lang + '_terms.tsv',
                          header=None, delimiter='\t')[0].tolist()
    elif ver == 'nes':
        df = pd.read_csv('../ACTER/'+ lang + '/htfl/annotated/annotations/unique_annotation_lists/htfl_'+ lang + '_terms_nes.tsv',
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
            en_ann_output1 = []
            for x, y in list(zip(df.words, df.en_ann_output1)):
                words = eval(x)
                labels = y[:len(words)*2]
                if labels.startswith("'I"):
                    labels = 'O '*len(words)
                labels =  filter_text(labels).split(' ')
                en_ann_output1.append(labels)
            df['en_ann_output1_processed'] = en_ann_output1
        
            en_ann_output1_processed = []
            for x, y in zip(df.words, df.en_ann_output1_processed):
                terms =  extract_entities(eval(x), y)
                en_ann_output1_processed.extend(terms)

            _, _, _, precision, recall, fscore = computeTermEvalMetrics(en_ann_output1_processed, gold_list)
            print("Precision: " + str(precision))
            print("Recall: " + str(recall))
            print("F-score: " + str(fscore))

            ##########################################
            print("#"*50)
            print("#2. Extracted candidate term list")
            en_ann_output2 = []
            for x in df['en_ann_output2']:
                if x.startswith('[') and x.endswith(']') and x.count('[') == 1: 
                    if "Jehovah's Witness" in x:
                        en_ann_output2.extend(["18-year-old", "Jehovah's Witness", "sickle cell disease", "life-threatening anemia"])
                    else:
                        en_ann_output2.extend(eval(x))
            _, _, _, precision, recall, fscore = computeTermEvalMetrics(en_ann_output2, gold_list)
            print("Precision: " + str(precision))
            print("Recall: " + str(recall))
            print("F-score: " + str(fscore))

            ##########################################
            print("#"*50)
            print("#3. Masking terms")
            df['en_ann_output3_processed'] = [extract_between_markers(x) for x in df['en_ann_output3']]
            en_ann_output3_processed = []
            for x in df['en_ann_output3_processed']:
                en_ann_output3_processed.extend(x)
            
            _, _, _, precision, recall, fscore = computeTermEvalMetrics(en_ann_output3_processed, gold_list)
            print("Precision: " + str(precision))
            print("Recall: " + str(recall))
            print("F-score: " + str(fscore))

        elif args.ver == 'nes':
            print("English NES evaluation")

            ##########################################
            print("#"*50)
            print("#1. Extracted IOB format")
            en_nes_output1 = []
            for x, y in list(zip(df.words, df.en_nes_output1)):
                words = eval(x)
                labels = y[:len(words)*2]
                if labels.startswith("'I"):
                    labels = 'O '*len(words)
                labels =  filter_text(labels).split(' ')
                en_nes_output1.append(labels)
            df['en_nes_output1_processed'] = en_nes_output1

            en_nes_output1_processed = []
            for x, y in zip(df.words, df.en_nes_output1_processed):
                terms =  extract_entities(eval(x), y)
                en_nes_output1_processed.extend(terms)
                
            _, _, _, precision, recall, fscore = computeTermEvalMetrics(en_nes_output1_processed, gold_list)
            print("Precision: " + str(precision))
            print("Recall: " + str(recall))
            print("F-score: " + str(fscore))

            ##########################################
            print("#"*50)
            print("#2. Extracted candidate term list")
            en_nes_output2 = []
            for x in df['en_nes_output2']:
                if x.startswith('[') and x.endswith(']') and x.count('[') == 1:
                    # print(x)
                    if "E/e'septal" in x:
                        en_nes_output2.extend(["E/e'septal", "PCWP", "E/e'lateral", "E/e'mean"])
                    elif "['cutoff', 'predictive value', 'E/e' septal', 'negative likelihood ratio']" in x:
                        en_nes_output2.extend(["cutoff", "predictive value", "E/e' septal", "negative likelihood ratio"])
                    else:
                        en_nes_output2.extend(eval(x))
        
            _, _, _, precision, recall, fscore = computeTermEvalMetrics(en_nes_output2, gold_list)
            print("Precision: " + str(precision))
            print("Recall: " + str(recall))
            print("F-score: " + str(fscore))

            ##########################################
            print("#"*50)
            print("#3. Masking terms")
            df['en_nes_output3_processed'] = [extract_between_markers(x) for x in df['en_nes_output3']]
            en_nes_output3_processed = []
            for x in df['en_nes_output3_processed']:
                en_nes_output3_processed.extend(x)

            _, _, _, precision, recall, fscore = computeTermEvalMetrics(en_nes_output3_processed, gold_list)
            print("Precision: " + str(precision))
            print("Recall: " + str(recall))
            print("F-score: " + str(fscore))
        else: 
            raise Exception("Version not supported")
    elif args.lang == 'fr':
        
        if args.ver == 'ann':
            print("French ANN evaluation")

            ##########################################
            print("#"*50)
            print("#1. Extracted IOB format")
            fr_ann_output1 = []
            for x, y in list(zip(df.words, df.fr_ann_output1)):
                if x == 'nan':
                    words = []
                else:
                    words = eval(x)
                labels = y[:len(words)*2]
                if labels.startswith("'I"):
                    labels = 'O '*len(words)
                labels =  filter_text(labels).split(' ')
                fr_ann_output1.append(labels)

            df['fr_ann_output1_processed'] = fr_ann_output1

            fr_ann_output1_processed = []
            for x, y in zip(df.words, df.fr_ann_output1_processed):
                terms =  extract_entities(eval(x), y)
                fr_ann_output1_processed.extend(terms)

            _, _, _, precision, recall, fscore = computeTermEvalMetrics(fr_ann_output1_processed, gold_list)
            print("Precision: " + str(precision))
            print("Recall: " + str(recall))
            print("F-score: " + str(fscore))

            ##########################################
            print("#"*50)
            print("#2. Extracted candidate term list")
            fr_ann_output2 = []
            for x in df['fr_ann_output2']:
                if x.startswith('[') and x.endswith(']') and x.count('[') == 1:
                    temp = x[2:-2].replace("', '", ',').split(',')
                    fr_ann_output2.extend(temp)
                    
            fr_ann_output2 = [x for x in fr_ann_output2 if len(x) != 0]

            _, _, _, precision, recall, fscore = computeTermEvalMetrics(fr_ann_output2, gold_list)
            print("Precision: " + str(precision))
            print("Recall: " + str(recall))
            print("F-score: " + str(fscore))

            ##########################################
            print("#"*50)
            print("#3. Masking terms")
            df['en_ann_output3_processed'] = [extract_between_markers(x) for x in df['fr_ann_output3']]
            en_ann_output3_processed = []
            for x in df['en_ann_output3_processed']:
                en_ann_output3_processed.extend(x)
            
            _, _, _, precision, recall, fscore = computeTermEvalMetrics(en_ann_output3_processed, gold_list)
            print("Precision: " + str(precision))
            print("Recall: " + str(recall))
            print("F-score: " + str(fscore))

        elif args.ver == 'nes':
            print("French NES evaluation")

            ##########################################
            print("#"*50)
            print("#1. Extracted candidate term list")
            fr_nes_output1 = []
            for x, y in list(zip(df.words, df.fr_nes_output1)):
                if x == 'nan':
                    words = []
                else:
                    words = eval(x)
                labels = y[:len(words)*2]
                if labels.startswith("'I"):
                    labels = 'O '*len(words)
                labels =  filter_text(labels).split(' ')
                fr_nes_output1.append(labels)

            df['fr_nes_output1_processed'] = fr_nes_output1

            fr_nes_output1_processed = []
            for x, y in zip(df.words, df.fr_nes_output1_processed):
                terms =  extract_entities(eval(x), y)
                fr_nes_output1_processed.extend(terms)
        
            _, _, _, precision, recall, fscore = computeTermEvalMetrics(fr_nes_output1_processed, gold_list)
            print("Precision: " + str(precision))
            print("Recall: " + str(recall))
            print("F-score: " + str(fscore))

            ##########################################
            print("#"*50)
            print("#2. Extracted candidate term list")
            
            fr_nes_output2 = []
            for x in df['fr_nes_output2']:
                if x.startswith('[') and x.endswith(']') and x.count('[') == 1:
                    temp = x[2:-2].replace("', '", ',').split(',')
                    fr_nes_output2.extend(temp)
                
            _, _, _, precision, recall, fscore = computeTermEvalMetrics(fr_nes_output2, gold_list)
            print("Precision: " + str(precision))
            print("Recall: " + str(recall))
            print("F-score: " + str(fscore))

            ##########################################
            print("#"*50)
            print("#3. Masking terms")
            df['fr_nes_output3_processed'] = [extract_between_markers(x) for x in df['fr_nes_output3']]
            fr_nes_output3_processed = []
            for x in df['fr_nes_output3_processed']:
                fr_nes_output3_processed.extend(x)

            _, _, _, precision, recall, fscore = computeTermEvalMetrics(fr_nes_output3_processed, gold_list)
            print("Precision: " + str(precision))
            print("Recall: " + str(recall))
            print("F-score: " + str(fscore))
        else: 
            raise Exception("Version not supported")
    elif args.lang == 'nl':
        
        if args.ver == 'ann':
            print("Dutch ANN evaluation")
            ##########################################
            print("#"*50)
            print("#1. Extracted IOB format")
            nl_ann_output1 = []
            count =  0
            for x, y in list(zip(df.words, df.nl_ann_output1)):
                # print(count)
                if str(x) == 'nan':
                    words = []
                    labels = ''
                else:
                    words = eval(x)
                    labels = y[:len(words)*2]
                if labels.startswith("'I"):
                    labels = 'O '*len(words)
                labels =  filter_text(labels).split(' ')
                nl_ann_output1.append(labels)
                count +=1

            df['nl_ann_output1_processed'] = nl_ann_output1

            nl_ann_output1_processed = []
            for x, y in zip(df.words, df.nl_ann_output1_processed):
                if str(x) != 'nan':
                    terms =  extract_entities(eval(x), y)
                    nl_ann_output1_processed.extend(terms)

            _, _, _, precision, recall, fscore = computeTermEvalMetrics(nl_ann_output1_processed, gold_list)
            print("Precision: " + str(precision))
            print("Recall: " + str(recall))
            print("F-score: " + str(fscore))

            ##########################################
            print("#"*50)
            print("#2. Extracted candidate term list")
            nl_ann_output2 = []
            for x in df['nl_ann_output2']:
                if str(x).startswith('[') and str(x).endswith(']') and str(x).count('[') == 1:
                    temp = x[2:-2].replace("', '", ',').split(',')
                    nl_ann_output2.extend(temp)
            nl_ann_output2 = [x for x in nl_ann_output2 if len(x) != 0]

            _, _, _, precision, recall, fscore = computeTermEvalMetrics(nl_ann_output2, gold_list)
            print("Precision: " + str(precision))
            print("Recall: " + str(recall))
            print("F-score: " + str(fscore))

            ##########################################
            print("#"*50)
            print("#3. Masking terms")
            df['nl_ann_output3_processed'] = [extract_between_markers(str(x)) for x in df['nl_ann_output3']]
            nl_ann_output3_processed = []
            for x in df['nl_ann_output3_processed']:
                nl_ann_output3_processed.extend(x)
            
            _, _, _, precision, recall, fscore = computeTermEvalMetrics(nl_ann_output3_processed, gold_list)
            print("Precision: " + str(precision))
            print("Recall: " + str(recall))
            print("F-score: " + str(fscore))

        elif args.ver == 'nes':
            print("Dutch NES evaluation")

            #########################################
            print("#"*50)
            print("#1. Extracted IOB format")
            # print(len(df))
            nl_nes_output1 = []
            count =  0
            for x, y in list(zip(df.words, df.nl_nes_output1)):
                # print(count, x, y)
                if str(x) == 'nan' or str(y) == 'nan':
                    words = []
                    labels = ''
                else:
                    words = eval(x)
                    labels = y[:len(words)*2]
                if labels.startswith("'I"):
                    labels = 'O '*len(words)
                labels =  filter_text(labels).split(' ')
                nl_nes_output1.append(labels)
                count +=1

            df['nl_nes_output1_processed'] = nl_nes_output1

            nl_nes_output1_processed = []
            for x, y in zip(df.words, df.nl_nes_output1_processed):
                if str(x) != 'nan':
                    terms =  extract_entities(eval(x), y)
                    nl_nes_output1_processed.extend(terms)
        
            _, _, _, precision, recall, fscore = computeTermEvalMetrics(nl_nes_output1_processed, gold_list)
            print("Precision: " + str(precision))
            print("Recall: " + str(recall))
            print("F-score: " + str(fscore))

            ##########################################
            print("#"*50)
            print("#2. Extracted candidate term list")
            nl_nes_output2 = []
            for x in df['nl_nes_output2']:
                if str(x).startswith('[') and str(x).endswith(']') and str(x).count('[') == 1:
                    temp = x[2:-2].replace("', '", ',').split(',')
                    nl_nes_output2.extend(temp)
            nl_nes_output2 = [x for x in nl_nes_output2 if len(x) != 0]
                
            _, _, _, precision, recall, fscore = computeTermEvalMetrics(nl_nes_output2, gold_list)
            print("Precision: " + str(precision))
            print("Recall: " + str(recall))
            print("F-score: " + str(fscore))

            ##########################################
            print("#"*50)
            print("#3. Masking terms")
            df['nl_nes_output3_processed'] = [extract_between_markers(str(x)) for x in df['nl_nes_output3']]
            nl_nes_output3_processed = []
            for x in df['nl_nes_output3_processed']:
                nl_nes_output3_processed.extend(x)

            _, _, _, precision, recall, fscore = computeTermEvalMetrics(nl_nes_output3_processed, gold_list)
            print("Precision: " + str(precision))
            print("Recall: " + str(recall))
            print("F-score: " + str(fscore))
        else: 
            raise Exception("Version not supported")
        
    else:
        raise Exception("Language not supported")

    
