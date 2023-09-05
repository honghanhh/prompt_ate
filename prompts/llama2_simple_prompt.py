import torch
import transformers
from transformers import AutoTokenizer
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate,  LLMChain

import argparse
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]='2'
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["WANDB_DISABLED"] = "true"


def prompt_design(lang, ver, format):
    if lang == 'en':
        if ver == 'ann':
            if format == 1:
                PROMPT = """
                As an excellent automatic term extraction (ATE) system, extract the terms in the Heart Failure domain given the following text delimited by triple backquotes. Named entities are not considered as terms.
                Output Format: IOB labeling for each word and punctuation where B stands for the beginning word in the term, I stands for the word inside the term, and O stands for the word not part of the term.
                
                Examples of the output format: 
                Sentence: 'Treatment of anemia in patients with heart disease : a clinical practice guideline from the American College of Physicians .'
                Domain: Heart failure
                Output: 'O O B O B O B I O O B I I O O O O O B O'
                Sentence: 'Recommendation 2 : ACP recommends against the use of erythropoiesis-stimulating agents in patients with mild to moderate anemia and congestive heart failure or coronary heart disease .'
                Domain: Heart failure
                Output: 'O O O O O O O O O B I O B O O O O B O B I I O B I I O'
                Sentence: 'Moreover , there is yet to be established a common consensus being used in current assays .'
                Domain: Heart failure
                Output: 'O O O O O O O O O O O O O O O O O'

                Sentence: ```{text}```
                Domain: Heart failure
                Output:
                """
            elif format == 2:
                PROMPT = """
                As an excellent automatic term extraction (ATE) system, extract the terms in the Heart Failure domain given the following text delimited by triple backquotes. Named entities are not considered as terms.
                Output Format: [list of terms present]
                If no terms are presented, keep it empty list: []
                
                Examples of the output format:
                Sentence: 'Treatment of anemia in patients with heart disease : a clinical practice guideline from the American College of Physicians .'
                Domain: Heart failure
                Output: ['anemia', 'patients', 'heart disease', 'clinical practice guideline', 'Physicians']
                Sentence: 'Recommendation 2 : ACP recommends against the use of erythropoiesis-stimulating agents in patients with mild to moderate anemia and congestive heart failure or coronary heart disease .'
                Domain: Heart failure
                Output: ['erythropoiesis-stimulating agents', 'patients', 'anemia', 'congestive heart failure', 'coronary heart disease']
                Sentence: 'Moreover , there is yet to be established a common consensus being used in current assays .'
                Domain: Heart failure
                Output: []

                Sentence: ```{text}```
                Domain: Heart failure
                Output: 
                """
            elif format == 3:
                PROMPT = """
                As an excellent automatic term extraction (ATE) system, extract the terms in the Heart Failure domain given the following text delimited by triple backquotes. Named entities are not considered as terms.
                Examples of the output format: 
                Sentence: 'Treatment of anemia in patients with heart disease : a clinical practice guideline from the American College of Physicians .'
                Domain: Heart failure
                Output: 'Treatment of @@anemia## in @@patients## with @@heart disease## : a @@clinical practice guideline## from the American College of @@Physicians## .'
                Sentence: 'Recommendation 2 : ACP recommends against the use of erythropoiesis-stimulating agents in patients with mild to moderate anemia and congestive heart failure or coronary heart disease .'
                Domain: Heart failure
                Output: 'Recommendation 2 : ACP recommends against the use of @@erythropoiesis-stimulating agents## in @@patients## with mild to moderate @@anemia## and @@congestive heart failure## or @@coronary heart disease## .'
                Sentence: 'Moreover , there is yet to be established a common consensus being used in current assays .'
                Domain: Heart failure
                Output: 'Moreover , there is yet to be established a common consensus being used in current assays .'

                Sentence: ```{text}```
                Domain: Heart failure
                Output:
                """
            else:
                raise Exception("Format not supported")
        elif ver == 'nes':
            if format == 1:
                PROMPT = """
                As an excellent automatic term extraction (ATE) system, extract the terms in the Heart Failure domain given the following text delimited by triple backquotes. Both terms and named entities are considered as terms.
                Output Format: IOB labeling for each word and punctuation where B stands for the beginning word in the term, I stands for the word inside the term, and O stands for the word not part of the term.
                
                Examples of the output format: 
                Sentence: 'Treatment of anemia in patients with heart disease : a clinical practice guideline from the American College of Physicians .'
                Domain: Heart failure
                Output: 'O O B O B O B I O O B I I O O B I I I O'
                Sentence: 'Recommendation 2 : ACP recommends against the use of erythropoiesis-stimulating agents in patients with mild to moderate anemia and congestive heart failure or coronary heart disease .'
                Domain: Heart failure
                Output: 'O O O B O O O O O B I O B O O O O B O B I I O B I I O'
                Sentence: 'Moreover , there is yet to be established a common consensus being used in current assays .'
                Domain: Heart failure
                Output: 'O O O O O O O O O O O O O O O O O'

                Sentence: ```{text}```
                Domain: Heart failure
                Output:
                """
            elif format == 2:
                PROMPT = """
                As an excellent automatic term extraction (ATE) system, extract the terms in the Heart Failure domain given the following text delimited by triple backquotes. Both terms and named entities are considered as terms.
                Output Format: [list of terms present]
                If no terms are presented, keep it empty list: []
                
                Examples of the output format:
                Sentence: 'Treatment of anemia in patients with heart disease : a clinical practice guideline from the American College of Physicians .'
                Domain: Heart failure
                Output: ['anemia', 'patients', 'heart disease', 'clinical practice guideline', 'American College of Physicians']
                Sentence: 'Recommendation 2 : ACP recommends against the use of erythropoiesis-stimulating agents in patients with mild to moderate anemia and congestive heart failure or coronary heart disease .'
                Domain: Heart failure
                Output: ['ACP', 'erythropoiesis-stimulating agents', 'patients', 'anemia', 'congestive heart failure', 'coronary heart disease']
                Sentence: 'Moreover , there is yet to be established a common consensus being used in current assays .'
                Domain: Heart failure
                Output: []

                Sentence: ```{text}```
                Domain: Heart failure
                Output: 
                """
            elif format == 3:
                PROMPT = """
                As an excellent automatic term extraction (ATE) system, extract the terms in the Heart Failure domain given the following text delimited by triple backquotes. Both terms and named entities are considered as terms.
                Examples of the output format: 
                Sentence: 'Treatment of anemia in patients with heart disease : a clinical practice guideline from the American College of Physicians .'
                Domain: Heart failure
                Output: 'Treatment of @@anemia## in @@patients## with @@heart disease## : a @@clinical practice guideline## from the @@American College of Physicians## .'
                Sentence: 'Recommendation 2 : ACP recommends against the use of erythropoiesis-stimulating agents in patients with mild to moderate anemia and congestive heart failure or coronary heart disease .'
                Domain: Heart failure
                Output: 'Recommendation 2 : @@ACP## recommends against the use of @@erythropoiesis-stimulating agents## in @@patients## with mild to moderate @@anemia## and @@congestive heart failure## or @@coronary heart disease## .'
                Sentence: 'Moreover , there is yet to be established a common consensus being used in current assays .'
                Domain: Heart failure
                Output: 'Moreover , there is yet to be established a common consensus being used in current assays .'

                Sentence: ```{text}```
                Domain: Heart failure
                Output:
                """
            else:
                raise Exception("Format not supported")
        else:
            raise Exception("Version not supported")
    elif lang == 'fr':
        if ver == 'ann':
            if format == 1:
                PROMPT = """
                As an excellent automatic term extraction (ATE) system, extract the terms in the Heart Failure domain given the following text delimited by triple backquotes. Named entities are not considered as terms.
                Output Format: IOB labeling for each word and punctuation where B stands for the beginning word in the term, I stands for the word inside the term, and O stands for the word not part of the term.
                
                Examples of the output format: 
                Sentence: 'Prévalence et prise en charge de l' insuffisance cardiaque en France : enquête nationale auprès des médecins généralistes du réseau Sentinelles .'
                Domain: Heart failure
                Output: 'B O B I I O O B I O O O O O O O B I O O O O'
                Sentence: 'Recommandations de la Conférence consensuelle de la Société canadienne de cardiologie 2006 sur l' insuffisance cardiaque : Diagnostic et prise en charge .'
                Domain: Heart failure
                Output: 'O O O O O O O O O O B O O O B I O B O B I I O'
                Sentence: 'La durée moyenne de séjour est de 11 jours .'
                Domain: Heart failure
                Output: 'O O O O O O O O O O'

                Sentence: ```{text}```
                Domain: Heart failure
                Output:
                """
            elif format == 2:
                PROMPT = """
                As an excellent automatic term extraction (ATE) system, extract the terms in the Heart Failure domain given the following text delimited by triple backquotes. Named entities are not considered as terms.
                Output Format: [list of terms present]
                If no terms are presented, keep it empty list: []
                
                Examples of the output format:  
                Sentence: 'Prévalence et prise en charge de l' insuffisance cardiaque en France : enquête nationale auprès des médecins généralistes du réseau Sentinelles .'
                Domain: Heart failure
                Output: ['Prévalence', 'prise en charge', 'insuffisance cardiaque', 'médecins généralistes']
                Sentence: 'Recommandations de la Conférence consensuelle de la Société canadienne de cardiologie 2006 sur l' insuffisance cardiaque : Diagnostic et prise en charge .'
                Domain: Heart failure
                Output: ['cardiologie', 'insuffisance cardiaque', 'Diagnostic', 'prise en charge']
                Sentence: 'La durée moyenne de séjour est de 11 jours .'
                Domain: Heart failure
                Output: []

                Sentence: ```{text}```
                Domain: Heart failure
                Output: """
            elif format == 3:
                PROMPT = """
                As an excellent automatic term extraction (ATE) system, extract the terms in the Heart Failure domain given the following text delimited by triple backquotes. Named entities are not considered as terms.
                Examples of the output format: 
                Sentence: 'Prévalence et prise en charge de l' insuffisance cardiaque en France : enquête nationale auprès des médecins généralistes du réseau Sentinelles .'
                Domain: Heart failure
                Output: '@@Prévalence## et @@prise en charge## de l' @@insuffisance cardiaque## en France : enquête nationale auprès des @@médecins généralistes## du réseau Sentinelles .'
                Sentence: 'Recommandations de la Conférence consensuelle de la Société canadienne de cardiologie 2006 sur l' insuffisance cardiaque : Diagnostic et prise en charge .'
                Domain: Heart failure
                Output: 'Recommandations de la Conférence consensuelle de la Société canadienne de @@cardiologie## 2006 sur l' @@insuffisance cardiaque## : @@Diagnostic## et @@prise en charge## .'
                Sentence: 'La durée moyenne de séjour est de 11 jours .'
                Domain: Heart failure
                Output: 'La durée moyenne de séjour est de 11 jours .'

                Sentence: ```{text}```
                Domain: Heart failure
                Output:
                """
            else:
                raise Exception("Format not supported")
        elif ver == 'nes':
            if format == 1:
                PROMPT = """
                As an excellent automatic term extraction (ATE) system, extract the terms in the Heart Failure domain given the following text delimited by triple backquotes. Both terms and named entities are considered as terms.
                Output Format: IOB labeling for each word and punctuation where B stands for the beginning word in the term, I stands for the word inside the term, and O stands for the word not part of the term.
                
                Examples of the output format: 
                Sentence: 'Prévalence et prise en charge de l' insuffisance cardiaque en France : enquête nationale auprès des médecins généralistes du réseau Sentinelles .'
                Domain: Heart failure
                Output: 'B O B I I O O B I O B O O O O O B I O B I O'
                Sentence: 'Recommandations de la Conférence consensuelle de la Société canadienne de cardiologie 2006 sur l' insuffisance cardiaque : Diagnostic et prise en charge .'
                Domain: Heart failure
                Output: 'O O O O O O O O O O B O O O B I O B O B I I O'
                Sentence: 'La durée moyenne de séjour est de 11 jours .'
                Domain: Heart failure
                Output: 'O O O O O O O O O O'

                Sentence: ```{text}```
                Domain: Heart failure
                Output:
                """
            elif format == 2:
                PROMPT = """
                As an excellent automatic term extraction (ATE) system, extract the terms in the Heart Failure domain given the following text delimited by triple backquotes. Both terms and named entities are considered as terms.
                Output Format: [list of terms present]
                If no terms are presented, keep it empty list: []
                
                Examples of the output format:  
                Sentence: 'Prévalence et prise en charge de l' insuffisance cardiaque en France : enquête nationale auprès des médecins généralistes du réseau Sentinelles .'
                Domain: Heart failure
                Output: ['Prévalence', 'prise en charge', 'insuffisance cardiaque', 'France', 'médecins généralistes', 'réseau Sentinelles']
                Sentence: 'Recommandations de la Conférence consensuelle de la Société canadienne de cardiologie 2006 sur l' insuffisance cardiaque : Diagnostic et prise en charge .'
                Domain: Heart failure
                Output: ['Conférence consensuelle de la Société canadienne de cardiologie', 'insuffisance cardiaque', 'Diagnostic', 'prise en charge']
                Sentence: 'La durée moyenne de séjour est de 11 jours .'
                Domain: Heart failure
                Output: []

                Sentence: ```{text}```
                Domain: Heart failure
                Output: 
                """
            elif format == 3:
                PROMPT = """
                As an excellent automatic term extraction (ATE) system, extract the terms in the Heart Failure domain given the following text delimited by triple backquotes. Both terms and named entities are considered as terms.
                Examples of the output format: 
                Sentence: 'Prévalence et prise en charge de l' insuffisance cardiaque en France : enquête nationale auprès des médecins généralistes du réseau Sentinelles .'
                Domain: Heart failure
                Output: '@@Prévalence## et @@prise en charge## de l' @@insuffisance cardiaque## en @@France## : enquête nationale auprès des @@médecins généralistes## du @@réseau Sentinelles## .'
                Sentence: 'Recommandations de la Conférence consensuelle de la Société canadienne de cardiologie 2006 sur l' insuffisance cardiaque : Diagnostic et prise en charge .'
                Domain: Heart failure
                Output: 'Recommandations de la @@Conférence consensuelle de la Société canadienne de cardiologie## 2006 sur l' @@insuffisance cardiaque## : @@Diagnostic## et @@prise en charge## .'
                Sentence: 'La durée moyenne de séjour est de 11 jours .'
                Domain: Heart failure
                Output: 'La durée moyenne de séjour est de 11 jours .'

                Sentence: ```{text}```
                Domain: Heart failure
                Output:
                """
            else:
                raise Exception("Format not supported")
        else:
            raise Exception("Version not supported")
    
    elif lang == 'nl':
        if ver == 'ann':
            if format == 1:
                PROMPT = """
                As an excellent automatic term extraction (ATE) system, extract the terms in the Heart Failure domain given the following text delimited by triple backquotes. Named entities are not considered as terms.
                Output Format: IOB labeling for each word and punctuation where B stands for the beginning word in the term, I stands for the word inside the term, and O stands for the word not part of the term.
                
                Examples of the output format: 
                Sentence: 'De bevindingen kunnen een grote rol spelen bij de herziening van de standaard Hartfalen van het Nederlands Huisartsen Genootschap .'
                Domain: Heart failure
                Output: 'O O O O O O O O O O O O O B O O O B O O'
                Sentence: 'Methode De reviewers zochten tot 2004 in MEDLINE , EMBASE , HERDIN en de Cochrane Library naar RCT's bij patiënten met chronisch hartfalen waarbij ( lis- of thiazide ) diuretica werden vergeleken met placebo of andere medicatie .'
                Domain: Heart failure
                Output: 'O O O O O O O O O O O O O O O O O B O B O B I O O B O B O B O O O B O O B O'
                Sentence: 'Na 1 nacht vasten werden lichaamsgewicht en vitale tekenen genoteerd .'
                Domain: Heart failure
                Output: 'O O O O O O O O O O O'

                Sentence: ```{text}```
                Domain: Heart failure
                Output:
                """
            elif format == 2:
                PROMPT = ""
            elif format == 3:
                PROMPT = """
                As an excellent automatic term extraction (ATE) system, extract the terms in the Heart Failure domain given the following text delimited by triple backquotes. Named entities are not considered as terms.
                Examples of the output format: 
                Sentence: 'De bevindingen kunnen een grote rol spelen bij de herziening van de standaard Hartfalen van het Nederlands Huisartsen Genootschap .'
                Domain: Heart failure
                Output: 'De bevindingen kunnen een grote rol spelen bij de herziening van de standaard @@Hartfalen## van het Nederlands @@Huisartsen## Genootschap .'
                Sentence: 'Methode De reviewers zochten tot 2004 in MEDLINE , EMBASE , HERDIN en de Cochrane Library naar RCT's bij patiënten met chronisch hartfalen waarbij ( lis- of thiazide ) diuretica werden vergeleken met placebo of andere medicatie .'
                Domain: Heart failure
                Output: 'Methode De reviewers zochten tot 2004 in MEDLINE , EMBASE , HERDIN en de Cochrane Library naar @@RCT's## bij @@patiënten## met @@chronisch hartfalen## waarbij ( @@lis-## of @@thiazide## ) @@diuretica## werden vergeleken met @@placebo## of andere @@medicatie## .'
                Sentence: 'Na 1 nacht vasten werden lichaamsgewicht en vitale tekenen genoteerd .'
                Domain: Heart failure
                Output: 'Na 1 nacht vasten werden lichaamsgewicht en vitale tekenen genoteerd .'

                Sentence: ```{text}```
                Domain: Heart failure
                Output:
                """
            else:
                raise Exception("Format not supported")
        elif ver == 'nes':
            if format == 1:
                PROMPT = """
                As an excellent automatic term extraction (ATE) system, extract the terms in the Heart Failure domain given the following text delimited by triple backquotes. Both terms and named entities are considered as terms.
                Output Format: IOB labeling for each word and punctuation where B stands for the beginning word in the term, I stands for the word inside the term, and O stands for the word not part of the term.
                
                Examples of the output format: 
                Sentence: 'De bevindingen kunnen een grote rol spelen bij de herziening van de standaard Hartfalen van het Nederlands Huisartsen Genootschap .'
                Domain: Heart failure
                Output: 'O O O O O O O O O O O O O B O O B I I O'
                Sentence: 'Methode De reviewers zochten tot 2004 in MEDLINE , EMBASE , HERDIN en de Cochrane Library naar RCT's bij patiënten met chronisch hartfalen waarbij ( lis- of thiazide ) diuretica werden vergeleken met placebo of andere medicatie .'
                Domain: Heart failure
                Output: 'O O O O O O O B O B O B O O B I O B O B O B I O O B O B O B O O O B O O B O'
                Sentence: 'Na 1 nacht vasten werden lichaamsgewicht en vitale tekenen genoteerd .'
                Domain: Heart failure
                Output: 'O O O O O O O O O O O'

                Sentence: ```{text}```
                Domain: Heart failure
                Output:
                """
            elif format == 2:
                PROMPT = ""
            elif format == 3:
                PROMPT = """
                As an excellent automatic term extraction (ATE) system, extract the terms in the Heart Failure domain given the following text delimited by triple backquotes. Both terms and named entities are considered as terms.
                Examples of the output format: 
                Sentence: 'De bevindingen kunnen een grote rol spelen bij de herziening van de standaard Hartfalen van het Nederlands Huisartsen Genootschap .'
                Domain: Heart failure
                Output: 'De bevindingen kunnen een grote rol spelen bij de herziening van de standaard @@Hartfalen## van het @@Nederlands Huisartsen Genootschap## .'
                Sentence: 'Methode De reviewers zochten tot 2004 in MEDLINE , EMBASE , HERDIN en de Cochrane Library naar RCT's bij patiënten met chronisch hartfalen waarbij ( lis- of thiazide ) diuretica werden vergeleken met placebo of andere medicatie .'
                Domain: Heart failure
                Output: 'Methode De reviewers zochten tot 2004 in @@MEDLINE## , @@EMBASE## , @@HERDIN## en de @@Cochrane Library## naar @@RCT's## bij @@patiënten## met @@chronisch hartfalen## waarbij ( @@lis-## of @@thiazide## ) @@diuretica## werden vergeleken met @@placebo## of andere @@medicatie## .'
                Sentence: 'Na 1 nacht vasten werden lichaamsgewicht en vitale tekenen genoteerd .'
                Domain: Heart failure
                Output: 'Na 1 nacht vasten werden lichaamsgewicht en vitale tekenen genoteerd .'

                Sentence: ```{text}```
                Domain: Heart failure
                Output:
                """
            else:
                raise Exception("Format not supported")
        else:
            raise Exception("Version not supported")
    
    else:
        raise Exception("Language not supported")
    
    return PROMPT

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="language", type=str, required=True)
    parser.add_argument("--ver", default="version", type=str, required=True)
    parser.add_argument("--formats", default="format", type=str, required=True)
    parser.add_argument("--output_path", default="output_path", type=str, required=True)
    args = parser.parse_args()

    if args.lang == 'en':
        htfl = pd.read_csv('../data/processed_data/en_htfl.csv')
    elif args.lang == 'fr':
        htfl = pd.read_csv('../data/processed_data/htfl.csv')
    elif args.lang == 'nl':
        htfl = pd.read_csv('../data/processed_data/nl_htfl.csv')
    else:
        raise Exception("Language not supported")

    model = "meta-llama/Llama-2-7b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(model)

    pipeline = transformers.pipeline(
        "text-generation", #task
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        max_length=1000,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )
    
    llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})

    template = prompt_design(args.lang, args.ver, args.formats)
    prompt = PromptTemplate(template=template, input_variables=["text"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    count = 0
    htfl['llama_output'+ str(args.formats) + args.ver] = pd.Series()
    for i in range(len(htfl)):
        htfl['llama_output3_ann'].iloc[i] = llm_chain.run(htfl['text'].iloc[i])
        if count % 50 == 0:
            print(str(count) + '/' + str(len(htfl)))
            print(htfl['text'].iloc[i])
            print(htfl['llama_output'+ str(args.formats) + args.ver].iloc[i])
        count +=1
    
    htfl.to_csv(args.output_path, index=False)
