from utils_metrics import get_entities_bio, f1_score, classification_report
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer
from transformers import MBartConfig, MBartForConditionalGeneration, MBart50Tokenizer
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'
import time
import math
import pandas as pd
import evaluate
metric = evaluate.load("seqeval")

class InputExample():
    def __init__(self, words, labels):
        self.words = words
        self.labels = labels

def template_entity(words, encoder_output, start):
    start_time = time.time()
    LABELS=['term']
    template_list=[" is a term"]
    entity_dict={i:e for i, e in enumerate(LABELS)}
    num_entities = len(template_list)
    
    # input text -> template
    words_length = len(words)
    words_length_list = [len(i) for i in words]

    encoder_output = encoder_output.repeat(num_entities*words_length, 1, 1)
    # print(encoder_output.shape)
    temp_list = []
    for i in range(words_length):
        for j in range(len(template_list)):
            temp_list.append(words[i]+template_list[j])

    # print("temp_list: ", temp_list)
    output_ids = tokenizer(temp_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
    # print("Before: ",output_ids.shape)
    output_ids[:, 0] = 2
    # print("After: ",output_ids.shape)
    output_length_list = [0]*num_entities*words_length

    for i in range(len(temp_list)//num_entities):
        base_length = ((tokenizer(temp_list[i * num_entities], return_tensors='pt', padding=True, truncation=True)['input_ids']).shape)[1] - 4
        output_length_list[i*num_entities:i*num_entities+ num_entities] = [base_length]*num_entities
        output_length_list[i*num_entities+4] += 1

    score = [1]*num_entities*words_length

    with torch.no_grad():
        decoder_output = decoder(input_ids=output_ids[:, :output_ids.shape[1] - 2].to(device), encoder_hidden_states=encoder_output)
        output = decoder_output.last_hidden_state
        
        lm_logits = model.lm_head(output)
        lm_logits = lm_logits + model.final_logits_bias.to(lm_logits.device)
        # print(lm_logits.shape)

        output = lm_logits

    for i in range(output_ids.shape[1] - 3):
        # print(input_ids.shape)
        logits = output[:, i, :]
        logits = logits.softmax(dim=1)
        # values, predictions = logits.topk(1,dim = 1)
        logits = logits.to('cpu').numpy()
        # print(output_ids[:, i+1].item())
        for j in range(0, num_entities*words_length):
            if i < output_length_list[j]:
                score[j] = score[j] * logits[j][int(output_ids[j][i + 1])]

    # print(cal_time(start_time))
    end = start+(score.index(max(score))//num_entities)
        # score_list.append(score)
    # print("score: ", score)
    return [start, end, entity_dict[(score.index(max(score))%num_entities)]if round(max(score),4) > 0 else 'O' , round(max(score),4)] #[start_index,end_index,label,score]

def prediction(input_TXT):
    # print("prediction")
    # print(input_TXT)
    
    input_TXT_list = input_TXT.split(' ')
    # print(len(input_TXT_list))

    input_TXT = [input_TXT]
    input_ids = tokenizer(input_TXT, return_tensors='pt')['input_ids']
    with torch.no_grad():
        encoder_output = encoder(input_ids=input_ids.to(device))[0]
        
    entity_list = []
    for i in range(len(input_TXT_list)):
        words = []
        for j in range(1, min(WORD_MAX_LENGTH+1, len(input_TXT_list) - i + 1)):
            word = (' ').join(input_TXT_list[i:i+j])
            words.append(word)

# print(words)
        entity = template_entity(words, encoder_output, i) #[start_index,end_index,label,score]
        # print("entity: ", entity)
        if entity[1] >= len(input_TXT_list):
            entity[1] = len(input_TXT_list)-1
        if entity[2] != 'O':
            entity_list.append(entity)
    i = 0
    if len(entity_list) > 1:
        while i < len(entity_list):
            j = i+1
            while j < len(entity_list):
                if (entity_list[i][1] < entity_list[j][0]) or (entity_list[i][0] > entity_list[j][1]):
                    j += 1
                else:
                    if entity_list[i][3] < entity_list[j][3]:
                        entity_list[i], entity_list[j] = entity_list[j], entity_list[i]
                        entity_list.pop(j)
                    else:
                        entity_list.pop(j)
            i += 1
    label_list = ['O'] * len(input_TXT_list)

    for entity in entity_list:
        label_list[entity[0]:entity[1]+1] = ["I-"+entity[2]]*(entity[1]-entity[0]+1)
        label_list[entity[0]] = "B-"+entity[2]
    return label_list

def cal_time(since):
    now = time.time()
    s = now - since
    ms = math.floor((s - math.floor(s)) * 1000)
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds %dms' % (m, s, ms)

MODEL_PATH = './outputs/best_multi'
WORD_MAX_LENGTH = 4
tokenizer = MBart50Tokenizer.from_pretrained('facebook/mbart-large-50')
model = MBartForConditionalGeneration.from_pretrained(MODEL_PATH)
model.eval()
model.config.use_cache = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

encoder = model.get_encoder()
decoder = model.get_decoder()


score_list = []
file_path = '/home/tranthh/semeval2023/multi_dev.conll'
guid_index = 1
examples = []
with open(file_path, "r", encoding="utf-8") as f:
    words = []
    labels = []
    for line in f:
        if line.startswith("-DOCSTART-") or line.startswith('#') or line == "" or line == "\n":
            if words:
                examples.append(InputExample(words=words, labels=labels))
                words = []
                labels = []
        else:
            splits = line.split(" ")
            words.append(splits[0])
            if len(splits) > 1:
                labels.append(splits[-1].replace("\n", ""))
            else:
                # Examples could have no label for mode = "test"
                labels.append("O")
    if words:
        examples.append(InputExample(words=words, labels=labels))
                
        
trues_list = []
preds_list = []
num_01 = len(examples)
num_point = 0
start = time.time()
for example in examples:
    sources = ' '.join(example.words)
    preds_list.append(prediction(sources))
    trues_list.append(example.labels)
    if num_point % 10 == 0:
        print('%d/%d (%s)'%(num_point+1, num_01, cal_time(start)))
        print(example.words)
        print('Pred:', preds_list[num_point])
        # print('Gold:', trues_list[num_point])
    num_point += 1
    
    
# true_entities = get_entities_bio(trues_list)
# pred_entities = get_entities_bio(preds_list)
# results = {
#     "f1": f1_score(true_entities, pred_entities)
# }
# print(classification_report(true_entities,pred_entities))

for num_point in range(len(preds_list)):
    preds_list[num_point] = ' '.join(preds_list[num_point]) + '\n'
    # trues_list[num_point] = ' '.join(trues_list[num_point]) + '\n'

final_preds = []
for x in preds_list:
    final_preds.extend([' '] + x.split() + [' '])
pd.DataFrame(final_preds).to_csv('./multi_dev.conll',  header=None,  index=False)