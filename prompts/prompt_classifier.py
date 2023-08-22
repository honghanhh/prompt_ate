import openai
import time
import pandas as pd
import argparse

def openai_chat_completion_response(final_prompt):
    response = openai.ChatCompletion.create(
              model="gpt-3.5-turbo",
              messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT_1},
                    {"role": "assistant", "content": ASSISTANT_PROMPT_1},
                    {"role": "user", "content": final_prompt}
                ]
            )

    return response['choices'][0]['message']['content'].strip(" \n")

def prompt_design(lang, ver, format):
    SYSTEM_PROMPT = "You are an excellent automatic term extraction (ATE) system. I will provide you the domain of the terms you need to extract and the sentence from which you need to extract the terms and the output in the given format with examples."
    USER_PROMPT_1 = "Are you clear about your role?"
    ASSISTANT_PROMPT_1 = "Sure, I'm ready to help you with your ATE task. Please provide me with the necessary information to get started."
    if lang == 'en':
        if ver == 'ann':
            if format == 1:
                PROMPT = ("What are the terms in the following text? Named entities are not considered as terms.\n"
                          "Output Format: IOB labeling for each word and punctuation where B stands for the beginning word in the term, I stands for the word inside the term, and O stands for the word not part of the term.\n"
                          "\n"
                          "Examples:\n"
                          "\n"
                          "Sentence:'Treatment of anemia in patients with heart disease : a clinical practice guideline from the American College of Physicians .'\n"
                          "Domain: Heart failure\n"
                          "Output: 'O O B O B O B I O O B I I O O O O O B O'\n"
                          "\n"
                          "Sentence: 'Recommendation 2 : ACP recommends against the use of erythropoiesis-stimulating agents in patients with mild to moderate anemia and congestive heart failure or coronary heart disease .'\n"
                          "Domain: Heart failure\n"
                          "Output: 'O O O O O O O O O B I O B O O O O B O B I I O B I I O'\n"
                          "\n"
                          "Sentence: 'Moreover , there is yet to be established a common consensus being used in current assays .'\n"
                          "Domain: Heart failure\n"
                          "Output: 'O O O O O O O O O O O O O O O O O'\n"
                          "\n"
                          "Sentence: {}\n"
                          "Domain: {}\n" 
                          "Output: "
                          )
            elif format == 2:
                PROMPT = ("What are the terms in the following text? Named entities are not considered as terms.\n"
                        "Output Format: [list of terms present]\n"
                        "If no terms are presented, keep it empty list: []\n"
                        "\n"
                        "Examples:\n"
                        "\n"
                        "Sentence:'Treatment of anemia in patients with heart disease : a clinical practice guideline from the American College of Physicians .'\n"
                        "Domain: Heart failure\n"
                        "Output: ['anemia', 'patients', 'heart disease', 'clinical practice guideline', 'Physicians']\n"
                        "\n"
                        "Sentence: 'Recommendation 2 : ACP recommends against the use of erythropoiesis-stimulating agents in patients with mild to moderate anemia and congestive heart failure or coronary heart disease .'\n"
                        "Domain: Heart failure\n"
                        "Output: ['erythropoiesis-stimulating agents', 'patients', 'anemia', 'congestive heart failure', 'coronary heart disease']\n"
                        "\n"
                        "Sentence: 'Moreover, there is yet to be established a common consensus being used in current assays .'\n"
                        "Domain: Heart failure\n"
                        "Output:[]\n"
                        "\n"
                        "Sentence: {}\n"
                        "Domain: {}\n"
                        "Output: "
                        )
            elif format == 3:
                PROMPT = ("What are the terms in the following text? Named entities are not considered as terms.\n"
                          "\n"
                          "Examples:\n"
                          "\n"
                          "Sentence:'Treatment of anemia in patients with heart disease : a clinical practice guideline from the American College of Physicians .'\n"
                          "Domain: Heart failure\n"
                          "Output: 'Treatment of @@anemia## in @@patients## with @@heart disease## : a @@clinical practice guideline## from the American College of @@Physicians## .'\n"
                          "\n"
                          "Sentence: 'Recommendation 2 : ACP recommends against the use of erythropoiesis-stimulating agents in patients with mild to moderate anemia and congestive heart failure or coronary heart disease .'\n"
                          "Domain: Heart failure\n"
                          "Output: 'Recommendation 2 : ACP recommends against the use of @@erythropoiesis-stimulating agents## in @@patients## with mild to moderate @@anemia## and @@congestive heart failure## or @@coronary heart disease## .'\n"
                          "\n"
                          "Sentence: 'Moreover , there is yet to be established a common consensus being used in current assays .'\n"
                          "Domain: Heart failure\n"
                          "Output: 'Moreover , there is yet to be established a common consensus being used in current assays .'\n"
                          "\n"
                          "Sentence: {}\n"
                          "Domain: {}\n"
                          "Output: "
                          )
            else:
                raise Exception("Format not supported")
        elif ver == 'nes':
            if format == 1:
                PROMPT = ("What are the terms in the following text? Both terms and named entities are considered as terms.\n"
                          "Output Format: IOB labeling for each word and punctuation where B stands for the beginning word in the term, I stands for the word inside the term, and O stands for the word not part of the term.\n"
                          "\n"
                          "Examples:\n"
                          "\n"
                          "Sentence:'Treatment of anemia in patients with heart disease : a clinical practice guideline from the American College of Physicians .'\n"
                          "Domain: Heart failure\n"
                          "Output: 'O O B O B O B I O O B I I O O B I I I O'\n"
                          "\n"
                          "Sentence: 'Recommendation 2 : ACP recommends against the use of erythropoiesis-stimulating agents in patients with mild to moderate anemia and congestive heart failure or coronary heart disease .'\n"
                          "Domain: Heart failure\n"
                          "Output: 'O O O B O O O O O B I O B O O O O B O B I I O B I I O'\n"
                          "\n"
                          "Sentence: 'Moreover , there is yet to be established a common consensus being used in current assays .'\n"
                          "Domain: Heart failure\n"
                          "Output: 'O O O O O O O O O O O O O O O O O'\n"
                          "\n"
                          "Sentence: {}\n"
                          "Domain: {}\n"
                          "Output: "
                        )
            elif format == 2:
                PROMPT = ("What are the terms in the following text? Both terms and named entities are considered as terms.\n"
                          "Output Format: [list of terms present]\n"
                          "If no terms are presented, keep it empty list: []\n"
                          "\n"
                          "Examples:\n"
                          "\n"
                          "Sentence:'Treatment of anemia in patients with heart disease : a clinical practice guideline from the American College of Physicians .'\n"
                          "Domain: Heart failure\n"
                          "Output: ['anemia', 'patients', 'heart disease', 'clinical practice guideline', 'American College of Physicians']\n"
                          "\n"
                          "Sentence: 'Recommendation 2 : ACP recommends against the use of erythropoiesis-stimulating agents in patients with mild to moderate anemia and congestive heart failure or coronary heart disease .'\n"
                          "Domain: Heart failure\n"
                          "Output: ['ACP', 'erythropoiesis-stimulating agents', 'patients', 'anemia', 'congestive heart failure', 'coronary heart disease']\n"
                          "\n"
                          "Sentence: 'Moreover, there is yet to be established a common consensus being used in current assays .'\n"
                          "Domain: Heart failure\n"
                          "Output:[]\n"
                          "\n"
                          "Sentence: {}\n"
                          "Domain: {}\n"
                          "Output: "
                          )
            elif format == 3:
                PROMPT = ("What are the terms in the following text? Both terms and named entities are considered as terms.\n"
                          "\n"
                          "Examples:\n"
                          "\n"
                          "Sentence:'Treatment of anemia in patients with heart disease : a clinical practice guideline from the American College of Physicians .'\n"
                          "Domain: Heart failure\n"
                          "Output: 'Treatment of @@anemia## in @@patients## with @@heart disease## : a @@clinical practice guideline## from the @@American College of Physicians## .\n"
                          "\n"
                          "Sentence: 'Recommendation 2 : ACP recommends against the use of erythropoiesis-stimulating agents in patients with mild to moderate anemia and congestive heart failure or coronary heart disease .'\n"
                          "Domain: Heart failure\n"
                          "Output: 'Recommendation 2 : @@ACP## recommends against the use of @@erythropoiesis-stimulating agents## in @@patients## with mild to moderate @@anemia## and @@congestive heart failure## or @@coronary heart disease## .'\n"
                          "\n"
                          "Sentence: 'Moreover, there is yet to be established a common consensus being used in current assays .'\n"
                          "Domain: Heart failure\n"
                          "Output: 'Moreover, there is yet to be established a common consensus being used in current assays .'\n"
                          "\n"
                          "Sentence: {}\n"
                          "Domain: {}\n"
                          "Output: "
                        )
            else:
                raise Exception("Format not supported")
        else:
            raise Exception("Version not supported")
    elif lang == 'fr':
        if ver == 'ann':
            if format == 1:
                PROMPT = ("What are the terms in the following text? Named entities are not considered as terms.\n"
                          "Output Format: IOB labeling for each word and punctuation where B stands for the beginning word in the term, I stands for the word inside the term, and O stands for the word not part of the term.\n"
                          "\n"
                          "Examples:\n"
                          "\n"
                          "Sentence:'Prévalence et prise en charge de l' insuffisance cardiaque en France : enquête nationale auprès des médecins généralistes du réseau Sentinelles .'\n"
                          "Domain: Heart failure\n"
                          "Output: 'B O B I I O O B I O O O O O O O B I O O O O'\n"
                          "\n"
                          "Sentence: 'Recommandations de la Conférence consensuelle de la Société canadienne de cardiologie 2006 sur l' insuffisance cardiaque : Diagnostic et prise en charge .'\n"
                          "Domain: Heart failure\n"
                          "Output: 'O O O O O O O O O O B O O O B I O B O B I I O'\n"
                          "\n"
                          "Sentence: 'La durée moyenne de séjour est de 11 jours .'\n"
                          "Domain: Heart failure\n"
                          "Output: 'O O O O O O O O O O'\n"
                          "\n"
                          "Sentence: {}\n"
                          "Domain: {}\n"
                          "Output: "
                        )
            elif format == 2:
                PROMPT = ("What are the terms in the following text? Named entities are not considered as terms.\n"
                          "Output Format: [list of terms present]\n"
                          "If no terms are presented, keep it empty list: []\n"
                          "\n"
                          "Examples:\n"
                          "\n"
                          "Sentence:'Prévalence et prise en charge de l' insuffisance cardiaque en France : enquête nationale auprès des médecins généralistes du réseau Sentinelles .'\n"
                          "Domain: Heart failure\n"
                          "Output: ['Prévalence', 'prise en charge', 'insuffisance cardiaque', 'médecins généralistes']\n"
                          "\n"
                          "Sentence: 'Recommandations de la Conférence consensuelle de la Société canadienne de cardiologie 2006 sur l' insuffisance cardiaque : Diagnostic et prise en charge .'\n"
                          "Domain: Heart failure\n"
                          "Output: ['cardiologie', 'insuffisance cardiaque', 'Diagnostic', 'prise en charge']\n"
                          "\n"
                          "Sentence: 'La durée moyenne de séjour est de 11 jours .'\n"
                          "Domain: Heart failure\n"
                          "Output:[]\n"
                          "\n"
                          "Sentence: {}\n"
                          "Domain: {}\n"
                          "Output: "
                          )
            elif format == 3:
                PROMPT = ("What are the terms in the following text? Named entities are not considered as terms.\n"
                          "\n"
                          "Examples:\n"
                          "\n"
                          "Sentence:'Prévalence et prise en charge de l' insuffisance cardiaque en France : enquête nationale auprès des médecins généralistes du réseau Sentinelles .'\n"
                          "Domain: Heart failure\n"
                          "Output: @@Prévalence## et @@prise en charge## de l' @@insuffisance cardiaque## en France : enquête nationale auprès des @@médecins généralistes## du réseau Sentinelles .'\n"
                          "\n"
                          "Sentence: 'Recommandations de la Conférence consensuelle de la Société canadienne de cardiologie 2006 sur l' insuffisance cardiaque : Diagnostic et prise en charge .'\n"
                          "Domain: Heart failure\n"
                          "Output: 'Recommandations de la Conférence consensuelle de la Société canadienne de @@cardiologie## 2006 sur l' @@insuffisance cardiaque## : @@Diagnostic## et @@prise en charge## .'\n"
                          "\n"
                          "Sentence: 'La durée moyenne de séjour est de 11 jours .'\n"
                          "Domain: Heart failure\n"
                          "Output: 'La durée moyenne de séjour est de 11 jours .'\n"
                          "\n"
                          "Sentence: {}\n"
                          "Domain: {}\n"
                          "Output: "
                        )
            else:
                raise Exception("Format not supported")
        elif ver == 'nes':
            if format == 1:
                PROMPT = ("What are the terms in the following text? Both terms and named entities are considered as terms.\n"
                          "\n"
                          "Examples:\n"
                          "\n"
                          "Sentence:'Prévalence et prise en charge de l' insuffisance cardiaque en France : enquête nationale auprès des médecins généralistes du réseau Sentinelles .'\n"
                          "Domain: Heart failure\n"
                          "Output: 'B O B I I O O B I O B O O O O O B I O B I'\n"
                          "\n"
                          "Sentence: 'Recommandations de la Conférence consensuelle de la Société canadienne de cardiologie 2006 sur l' insuffisance cardiaque : Diagnostic et prise en charge .'\n"
                          "Domain: Heart failure\n"
                          "Output: 'O O O B I I I I I I I O O O B I O B O B I I'\n"
                          "\n"
                          "Sentence: 'La durée moyenne de séjour est de 11 jours .'\n"
                          "Domain: Heart failure\n"
                          "Output: 'O O O O O O O O O O'\n"
                          "\n"
                          "Sentence: {}\n"
                          "Domain: {}\n"
                          "Output: "
                        )
            elif format == 2:
                PROMPT = ("What are the terms in the following text? Both terms and named entities are considered as terms.\n"
                          "Output Format: [list of terms present]\n"
                          "If no terms are presented, keep it empty list: []\n"
                          "\n"
                          "Examples:\n"
                          "\n"
                          "Sentence:'Prévalence et prise en charge de l' insuffisance cardiaque en France : enquête nationale auprès des médecins généralistes du réseau Sentinelles .'\n"
                          "Domain: Heart failure\n"
                          "Output: ['Prévalence', 'prise en charge', 'insuffisance cardiaque', 'France', 'médecins généralistes', 'réseau Sentinelles']\n"
                          "\n"
                          "Sentence: 'Recommandations de la Conférence consensuelle de la Société canadienne de cardiologie 2006 sur l' insuffisance cardiaque : Diagnostic et prise en charge .'\n"
                          "Domain: Heart failure\n"
                          "Output: ['Conférence consensuelle de la Société canadienne de cardiologie', 'insuffisance cardiaque', 'Diagnostic', 'prise en charge']\n"
                          "\n"
                          "Sentence: 'La durée moyenne de séjour est de 11 jours .'\n"
                          "Domain: Heart failure\n"
                          "Output: []\n"
                          "\n"
                          "Sentence: {}\n"
                          "Domain: {}\n"
                          "Output: "
                        )
            elif format == 3:
                PROMPT = ("What are the terms in the following text? Both terms and named entities are considered as terms.\n"
                          "\n"
                          "Examples:\n"
                          "\n"
                          "Sentence:'Prévalence et prise en charge de l' insuffisance cardiaque en France : enquête nationale auprès des médecins généralistes du réseau Sentinelles .'\n"
                          "Domain: Heart failure\n"
                          "Output: '@@Prévalence## et @@prise en charge## de l' @@insuffisance cardiaque## en @@France## : enquête nationale auprès des @@médecins généralistes## du @@réseau Sentinelles## .'\n"
                          "\n"
                          "Sentence: 'Recommandations de la Conférence consensuelle de la Société canadienne de cardiologie 2006 sur l' insuffisance cardiaque : Diagnostic et prise en charge .'\n"
                          "Domain: Heart failure\n"
                          "Output: 'Recommandations de la @@Conférence consensuelle de la Société canadienne de cardiologie## 2006 sur l' @@insuffisance cardiaque## : @@Diagnostic## et @@prise en charge## .'\n"
                          "\n"
                          "Sentence: 'La durée moyenne de séjour est de 11 jours .'\n"
                          "Domain: Heart failure\n"
                          "Output: 'La durée moyenne de séjour est de 11 jours .'\n"
                          "\n"
                          "Sentence: {}\n"
                          "Domain: {}\n"
                          "Output: "
                          )
            else:
                raise Exception("Format not supported")
        else:
            raise Exception("Version not supported")
    
    elif lang == 'nl':
        if ver == 'ann':
            if format == 1:
                PROMPT = ("What are the terms in the following text? Named entities are not considered as terms.\n"
                          "Output Format: IOB labeling for each word and punctuation where B stands for the beginning word in the term, I stands for the word inside the term, and O stands for the word not part of the term.\n"
                          "\n"
                          "Examples:\n"
                          "\n"
                          "Sentence: 'De bevindingen kunnen een grote rol spelen bij de herziening van de standaard Hartfalen van het Nederlands Huisartsen Genootschap .'\n"
                          "Domain: Heart failure\n"
                          "Output: 'O O O O O O O O O O O O O B O O O B O O'\n"
                          "\n"
                          "Sentence: 'Methode De reviewers zochten tot 2004 in MEDLINE , EMBASE , HERDIN en de Cochrane Library naar RCT's bij patiënten met chronisch hartfalen waarbij ( lis- of thiazide ) diuretica werden vergeleken met placebo of andere medicatie .'\n"
                          "Domain: Heart failure\n"
                          "Output: 'O O O O O O O O O O O O O O O O O B O B O B I O O B O B O B O O O B O O B O'\n"
                          "\n"
                          "Sentence: 'Na 1 nacht vasten werden lichaamsgewicht en vitale tekenen genoteerd .'\n"
                          "Domain: Heart failure\n"
                          "Output: 'O O O O O O O O O O O'\n"
                          "\n"
                          "Sentence: {}\n"
                          "Domain: {}\n"
                          "Output: "
                        )
            elif format == 2:
                PROMPT = ("What are the terms in the following text? Named entities are not considered as terms.\n"
                          "Output Format: [list of terms present]\n"
                          "If no terms are presented, keep it empty list: []\n"
                          "\n"
                          "Examples:\n"
                          "\n"
                          "Sentence: 'De bevindingen kunnen een grote rol spelen bij de herziening van de standaard Hartfalen van het Nederlands Huisartsen Genootschap .'\n"
                          "Domain: Heart failure\n"
                          "Output: ['Hartfalen', 'Huisartsen']\n"
                          "\n"
                          "Sentence: 'Methode De reviewers zochten tot 2004 in MEDLINE , EMBASE , HERDIN en de Cochrane Library naar RCT's bij patiënten met chronisch hartfalen waarbij ( lis- of thiazide ) diuretica werden vergeleken met placebo of andere medicatie .'\n""Domain: Heart failure\n"
                          "Output: ['RCT's', 'patiënten', 'chronisch hartfalen', 'lis-', 'thiazide', 'diuretica', 'placebo', 'medicatie']\n"
                          "\n"
                          "Sentence: 'Na 1 nacht vasten werden lichaamsgewicht en vitale tekenen genoteerd .'\n"
                          "Domain: Heart failure\n"
                          "Output: []\n"
                          "\n"
                          "Sentence: {}\n"
                          "Domain: {}\n"
                          "Output: "
                        )
            elif format == 3:
                PROMPT = ("What are the terms in the following text? Named entities are not considered as terms.\n"
                          "\n"
                          "Examples:\n"
                          "\n"
                          "Sentence: 'De bevindingen kunnen een grote rol spelen bij de herziening van de standaard Hartfalen van het Nederlands Huisartsen Genootschap .'\n"
                          "Domain: Heart failure\n"
                          "Output: 'De bevindingen kunnen een grote rol spelen bij de herziening van de standaard @@Hartfalen## van het Nederlands @@Huisartsen## Genootschap .'\n"
                          "\n"
                          "Sentence: 'Methode De reviewers zochten tot 2004 in MEDLINE , EMBASE , HERDIN en de Cochrane Library naar RCT's bij patiënten met chronisch hartfalen waarbij ( lis- of thiazide ) diuretica werden vergeleken met placebo of andere medicatie .'\n"
                          "Domain: Heart failure\n"
                          "Output: 'Methode De reviewers zochten tot 2004 in MEDLINE , EMBASE , HERDIN en de Cochrane Library naar @@RCT's## bij @@patiënten## met @@chronisch hartfalen## waarbij ( @@lis-## of @@thiazide## ) @@diuretica## werden vergeleken met @@placebo## of andere @@medicatie## .'\n"
                          "\n"
                          "Sentence: 'Na 1 nacht vasten werden lichaamsgewicht en vitale tekenen genoteerd .'\n"
                          "Domain: Heart failure\n"
                          "Output: 'Na 1 nacht vasten werden lichaamsgewicht en vitale tekenen genoteerd .'\n"
                          "\n"
                          "Sentence: {}\n"
                          "Domain: {}\n"
                          "Output: "
                          )
            else:
                raise Exception("Format not supported")
        elif ver == 'nes':
            if format == 1:
                PROMPT = ("What are the terms in the following text? Both terms and named entities are considered as terms.\n"
                          "Output Format: IOB labeling for each word and punctuation where B stands for the beginning word in the term, I stands for the word inside the term, and O stands for the word not part of the term.\n"
                          "\n"
                          "Examples:\n"
                          "\n"
                          "Sentence: 'De bevindingen kunnen een grote rol spelen bij de herziening van de standaard Hartfalen van het Nederlands Huisartsen Genootschap .'\n"
                          "Domain: Heart failure\n"
                          "Output: 'O O O O O O O O O O O O O B O O O B I I'\n"
                          "\n"
                          "Sentence: 'Methode De reviewers zochten tot 2004 in MEDLINE , EMBASE , HERDIN en de Cochrane Library naar RCT's bij patiënten met chronisch hartfalen waarbij ( lis- of thiazide ) diuretica werden vergeleken met placebo of andere medicatie .'\n"
                          "Domain: Heart failure\n"
                          "Output: 'O O O O O O O B O B O B O O B I O B O B O B I O O B O B O B O O O B O O B O'\n"
                          "\n"
                          "Sentence: 'Na 1 nacht vasten werden lichaamsgewicht en vitale tekenen genoteerd .'\n"
                          "Domain: Heart failure\n"
                          "Output: 'O O O O O O O O O O O'\n"
                          "\n"
                          "Sentence: {}\n"
                          "Domain: {}\n"
                          "Output: "
                          )
            elif format == 2:
                PROMPT = ("What are the terms in the following text? Both terms and named entities are considered as terms.\n"
                          "Output Format: [list of terms present]\n"
                          "If no terms are presented, keep it empty list: []\n"
                          "\n"
                          "Examples:\n"
                          "\n"
                          "Sentence: 'De bevindingen kunnen een grote rol spelen bij de herziening van de standaard Hartfalen van het Nederlands Huisartsen Genootschap .'\n"
                          "Domain: Heart failure\n"
                          "Output: ['Hartfalen', 'Nederlands Huisartsen Genootschap']\n"
                          "\n"
                          "Sentence: 'Methode De reviewers zochten tot 2004 in MEDLINE , EMBASE , HERDIN en de Cochrane Library naar RCT's bij patiënten met chronisch hartfalen waarbij ( lis- of thiazide ) diuretica werden vergeleken met placebo of andere medicatie .'\n"
                          "Domain: Heart failure\n"
                          "Output: ['MEDLINE', 'EMBASE', 'HERDIN', 'Cochrane Library', 'RCT's', 'patiënten', 'chronisch hartfalen', 'lis-', 'thiazide', 'diuretica', 'placebo', 'medicatie']\n"
                          "\n"
                          "Sentence: 'Na 1 nacht vasten werden lichaamsgewicht en vitale tekenen genoteerd .'\n"
                          "Domain: Heart failure\n"
                          "Output: []\n"
                          "\n"
                          "Sentence: {}\n"
                          "Domain: {}\n"
                          "Output: "
                          )
            elif format == 3:
                PROMPT = ("What are the terms in the following text? Both terms and named entities are considered as terms.\n"
                          "\n"
                          "Examples:\n"
                          "\n"
                          "Sentence: 'De bevindingen kunnen een grote rol spelen bij de herziening van de standaard Hartfalen van het Nederlands Huisartsen Genootschap .'\n"
                          "Domain: Heart failure\n"
                          "Output: 'De bevindingen kunnen een grote rol spelen bij de herziening van de standaard @@Hartfalen## van het @@Nederlands Huisartsen Genootschap## .'\n"
                          "\n"
                          "Sentence: 'Methode De reviewers zochten tot 2004 in MEDLINE , EMBASE , HERDIN en de Cochrane Library naar RCT's bij patiënten met chronisch hartfalen waarbij ( lis- of thiazide ) diuretica werden vergeleken met placebo of andere medicatie .'\n"
                          "Domain: Heart failure\n"
                          "Output: 'Methode De reviewers zochten tot 2004 in @@MEDLINE## , @@EMBASE## , @@HERDIN## en de @@Cochrane Library## naar @@RCT's## bij @@patiënten## met @@chronisch hartfalen## waarbij ( @@lis-## of @@thiazide## ) @@diuretica## werden vergeleken met @@placebo## of andere @@medicatie## .'\n"
                          "\n"
                          "Sentence: 'Na 1 nacht vasten werden lichaamsgewicht en vitale tekenen genoteerd .'\n"
                          "Domain: Heart failure\n"
                          "Output: 'Na 1 nacht vasten werden lichaamsgewicht en vitale tekenen genoteerd .'\n"
                          "\n"
                          "Sentence: {}\n"
                          "Domain: {}\n"
                          "Output: "
                        )
            else:
                raise Exception("Format not supported")
        else:
            raise Exception("Version not supported")
    
    else:
        raise Exception("Language not supported")
    
    return SYSTEM_PROMPT, USER_PROMPT_1, ASSISTANT_PROMPT_1, PROMPT



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data_path", type=str, required=True)
    parser.add_argument("--lang", default="language", type=str, required=True)
    parser.add_argument("--ver", default="version", type=str, required=True)
    parser.add_argument("--formats", default="format", type=str, required=True)
    parser.add_argument("--output_path", default="output_path", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)

    openai.api_key = ### YOUR API KEY HERE ###

    SYSTEM_PROMPT, USER_PROMPT_1, ASSISTANT_PROMPT_1, PROMPT = prompt_design(args.lang, args.ver, args.format)

    terms_list = []
    count = 0
    # text_length  = len(df.text.to_list())
    for x in df.text.to_list():
        # print("Sentence " + str(count) + '/' + str(text_length) + ' : '+ x)
        promt_temp = PROMPT.format(x, "Heart failure")
        term = openai_chat_completion_response(promt_temp)
        terms_list.append(term)
        # print("Terms :", term)
        if count % 60 == 0:
            time.sleep(60)
        count += 1
    
    df['preds_format' + args.formats] = terms_list
    df.to_csv(args.output_path, index=False)
