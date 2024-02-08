import torch
import transformers
from transformers import AutoTokenizer
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate,  LLMChain

from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["CUDA_VISIBLE_DEVICES"]='4'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"


def prompt_design(lang, ver, format):
    """
    1: cross-domain few-shot demonstration with specified domain
    2: in-domain few-shot demonstration with no specified domain
    3: cross-domain few-shot demonstration with no specified domain
    """
    if lang == 'en':
        if ver == 'ann':
            if format == 1:
                PROMPT = """
                <s>
                [INST]
                <<SYS>>
                As an excellent automatic term extraction (ATE) system, extract the terms in the Heart Failure domain given the following text delimited by triple backquotes. Named entities are not considered as terms.
                Output Format: [list of terms present]
                If no terms are presented, keep it empty list: []

                Examples of the output format:
                <</SYS>>
                Sentence: ```Conscious of its central position within the European Union , Belgium has , for many years , taken a firm line against corruption in national and international transactions .```
                Domain: Corruption
                [/INST]
                Output: ["corruption", "transactions"]
                </s>
                
                <s>
                [INST]
                Sentence: ```The Convention devotes a chapter to international cooperation .```
                Domain: Corruption
                [/INST]
                Output: []
                </s>
                
                <s>
                [INST]
                Sentence: ```The total installed capacity in the world as of the year 2001 was approximately 20.000 MW , with the majority of installations in Europe .```
                Domain: Wind energy
                [/INST]
                Output: ["installed capacity",  "MW"]
                </s>
                
                <s>
                [INST]
                Sentence: ```I hereby declare that all information in this document has been obtained and presented in accordance with academic rules and ethical conduct .```
                Domain: Wind energy
                [/INST]
                Output: []
                </s>
                
                <s>
                [INST]
                Sentence: ```There is some evidence that about 3,000 BC , near the Dnieper River and the Don River , people were using bits on horses , as a stallion that was buried there shows teeth wear consistent with using a bit .```
                Domain: Equitation
                [/INST]
                Output: ["bits", "horses", "stallion", "bit"]
                </s>
                
                <s>
                [INST]
                Sentence: ```In some parts of the world , they are still used for practical purposes such as farming .```
                Domain: Equitation
                [/INST]
                Output: []
                </s>
                
                <s>
                [INST]
                Sentence: ```{text}```
                Domain: Heart failure
                [/INST]
                """
            elif format == 2:
                PROMPT = """
                <s>
                [INST]
                <<SYS>>
                As an excellent automatic term extraction (ATE) system, extract the terms given the following text delimited by triple backquotes. Named entities are not considered as terms.
                Output Format: [list of terms present]
                If no terms are presented, keep it empty list: []

                Examples of the output format:
                <</SYS>>
                Sentence: ```Treatment of anemia in patients with heart disease : a clinical practice guideline from the American College of Physicians .```
                [/INST]
                Output: ["anemia", "patients", "heart disease", "clinical practice guideline", "Physicians"]
                </s>
                
                <s>
                [INST]
                Sentence: ```Recommendation 2 : ACP recommends against the use of erythropoiesis-stimulating agents in patients with mild to moderate anemia and congestive heart failure or coronary heart disease .```
                [/INST]
                Output: ["erythropoiesis-stimulating agents", "patients", "anemia", "congestive heart failure", "coronary heart disease"]
                </s>
                
                <s>
                [INST]
                Sentence: ```Moreover , there is yet to be established a common consensus being used in current assays .```
                [/INST]
                Output: []
                </s>
                
                <s>
                [INST]
                Sentence: ```{text}```
                [/INST]
                """
            elif format == 3:
                PROMPT = """
                <s>
                [INST]
                <<SYS>>
                As an excellent automatic term extraction (ATE) system, extract the terms given the following text delimited by triple backquotes. Named entities are not considered as terms.
                Output Format: [list of terms present]
                If no terms are presented, keep it empty list: []

                Examples of the output format:
                <</SYS>>
                Sentence: ```Conscious of its central position within the European Union , Belgium has , for many years , taken a firm line against corruption in national and international transactions .```
                [/INST]
                Output: ["corruption", "transactions"]
                </s>
                
                <s>
                [INST]
                Sentence: ```The Convention devotes a chapter to international cooperation .```
                [/INST]
                Output: []
                </s>
                
                <s>
                [INST]
                Sentence: ```The total installed capacity in the world as of the year 2001 was approximately 20.000 MW , with the majority of installations in Europe .```
                [/INST]
                Output: ["installed capacity",  "MW"]
                </s>
                
                <s>
                [INST]
                Sentence: ```I hereby declare that all information in this document has been obtained and presented in accordance with academic rules and ethical conduct .```
                [/INST]
                Output: []
                </s>
                
                <s>
                [INST]
                Sentence: ```There is some evidence that about 3,000 BC , near the Dnieper River and the Don River , people were using bits on horses , as a stallion that was buried there shows teeth wear consistent with using a bit .```
                [/INST]
                Output: ["bits", "horses", "stallion", "bit"]
                </s>
                
                <s>
                [INST]
                Sentence: ```In some parts of the world , they are still used for practical purposes such as farming .```
                [/INST]
                Output: []
                </s>
                
                <s>
                [INST]
                Sentence: ```{text}```
                [/INST]
                """
            else:
                raise Exception("Format not supported")
        elif ver == 'nes':
            if format == 1:
                PROMPT = 
                PROMPT = """
                <s>
                [INST]
                <<SYS>>
                As an excellent automatic term extraction (ATE) system, extract the terms in the Heart Failure domain given the following text delimited by triple backquotes. Named entities are not considered as terms.
                Output Format: [list of terms present]
                If no terms are presented, keep it empty list: []

                Examples of the output format:
                <</SYS>>
                Sentence: ```Conscious of its central position within the European Union , Belgium has , for many years , taken a firm line against corruption in national and international transactions .```
                Domain: Corruption
                [/INST]
                Output: ["European Union", "Belgium", "corruption", "transactions"]
                </s>
                
                <s>
                [INST]
                Sentence: ```The Convention devotes a chapter to international cooperation .```
                Domain: Corruption
                [/INST]
                Output: []
                </s>
                
                <s>
                [INST]
                Sentence: ```The total installed capacity in the world as of the year 2001 was approximately 20.000 MW , with the majority of installations in Europe .```
                Domain: Wind energy
                [/INST]
                Output: ["installed capacity",  "MW", "Europe"]
                </s>
                
                <s>
                [INST]
                Sentence: ```I hereby declare that all information in this document has been obtained and presented in accordance with academic rules and ethical conduct .```
                Domain: Wind energy
                [/INST]
                Output: []
                </s>
                
                <s>
                [INST]
                Sentence: ```There is some evidence that about 3,000 BC , near the Dnieper River and the Don River , people were using bits on horses , as a stallion that was buried there shows teeth wear consistent with using a bit .```
                Domain: Equitation
                [/INST]
                Output: ["Dnieper River", "Don River", "bits", "horses", "stallion", "bit"]
                </s>
                
                <s>
                [INST]
                Sentence: ```In some parts of the world , they are still used for practical purposes such as farming .```
                Domain: Equitation
                [/INST]
                Output: []
                </s>
                
                <s>
                [INST]
                Sentence: ```{text}```
                Domain: Heart failure
                [/INST]
                """
            elif format == 2:
                PROMPT = """
                <s>
                [INST]
                <<SYS>>
                As an excellent automatic term extraction (ATE) system, extract the terms given the following text delimited by triple backquotes. Both terms and named entities are considered as terms.
                Output Format: [list of terms present]
                If no terms are presented, keep it empty list: []

                Examples of the output format:
                <</SYS>>
                Sentence: ```Treatment of anemia in patients with heart disease : a clinical practice guideline from the American College of Physicians .```
                [/INST]
                Output: ["anemia", "patients", "heart disease", "clinical practice guideline", "American College of Physicians"]
                </s>
                
                <s>
                [INST]
                Sentence: ```Recommendation 2 : ACP recommends against the use of erythropoiesis-stimulating agents in patients with mild to moderate anemia and congestive heart failure or coronary heart disease .```
                [/INST]
                Output: ["ACP", "erythropoiesis-stimulating agents", "patients", "anemia", "congestive heart failure", "coronary heart disease"]
                </s>
                
                <s>
                [INST]
                Sentence: ```Moreover , there is yet to be established a common consensus being used in current assays .```
                [/INST]
                Output: []
                </s>
                
                <s>
                [INST]
                Sentence: ```{text}```
                [/INST]
                """

            elif format == 3:
                PROMPT = 
                PROMPT = """
                <s>
                [INST]
                <<SYS>>
                As an excellent automatic term extraction (ATE) system, extract the terms given the following text delimited by triple backquotes. Named entities are not considered as terms.
                Output Format: [list of terms present]
                If no terms are presented, keep it empty list: []

                Examples of the output format:
                <</SYS>>
                Sentence: ```Conscious of its central position within the European Union , Belgium has , for many years , taken a firm line against corruption in national and international transactions .```
                [/INST]
                Output: ["European Union", "Belgium", "corruption", "transactions"]
                </s>
                
                <s>
                [INST]
                Sentence: ```The Convention devotes a chapter to international cooperation .```
                [/INST]
                Output: []
                </s>
                
                <s>
                [INST]
                Sentence: ```The total installed capacity in the world as of the year 2001 was approximately 20.000 MW , with the majority of installations in Europe .```
                [/INST]
                Output: ["installed capacity",  "MW", "Europe"]
                </s>
                
                <s>
                [INST]
                Sentence: ```I hereby declare that all information in this document has been obtained and presented in accordance with academic rules and ethical conduct .```
                [/INST]
                Output: []
                </s>
                
                <s>
                [INST]
                Sentence: ```There is some evidence that about 3,000 BC , near the Dnieper River and the Don River , people were using bits on horses , as a stallion that was buried there shows teeth wear consistent with using a bit .```
                [/INST]
                Output: ["Dnieper River", "Don River", "bits", "horses", "stallion", "bit"]
                </s>
                
                <s>
                [INST]
                Sentence: ```In some parts of the world , they are still used for practical purposes such as farming .```
                [/INST]
                Output: []
                </s>
                
                <s>
                [INST]
                Sentence: ```{text}```
                [/INST]
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
    parser.add_argument("--formats", default="format", type=int, required=True)
    parser.add_argument("--output_path", default="output_path", type=str, required=True)
    parser.add_argument("--model", default="model", type=str, required=True)
    args = parser.parse_args()

    if args.lang == 'en':
        htfl = pd.read_csv('../data/processed_data/en_htfl.csv')
    elif args.lang == 'fr':
        htfl = pd.read_csv('../data/processed_data/fr_htfl.csv')
    elif args.lang == 'nl':
        htfl = pd.read_csv('../data/processed_data/nl_htfl.csv')
    else:
        raise Exception("Language not supported")


    dataset =  Dataset.from_pandas(htfl)

    model = args.model #"meta-llama/Llama-2-7b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(model)

    pipeline = transformers.pipeline(
        "text-generation", #task
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        #device_map="auto",
        device="cuda:1",
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

    htfl['llama_output'+ str(args.formats) + args.ver] = pd.Series()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    cumulative_out  = list()
    print(args.output_path)
    for batch in tqdm(iter(dataloader)):
        out_texts=[output["text"].strip().replace("\n","") for output in llm_chain.batch(batch["text"])]
        cumulative_out += out_texts

    htfl['llama_output'+ str(args.formats) + args.ver] = cumulative_out 
    
    htfl.to_csv(args.output_path, index=False)
    
