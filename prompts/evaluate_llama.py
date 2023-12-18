import re
import pandas as pd
import argparse

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


def read_data(lang, ver):
    if ver == "ann":
        df = pd.read_csv(
            "../ACTER/"
            + lang
            + "/htfl/annotated/annotations/unique_annotation_lists/htfl_"
            + lang
            + "_terms.tsv",
            header=None,
            delimiter="\t",
        )[0].tolist()
    elif ver == "nes":
        df = pd.read_csv(
            "../ACTER/"
            + lang
            + "/htfl/annotated/annotations/unique_annotation_lists/htfl_"
            + lang
            + "_terms_nes.tsv",
            header=None,
            delimiter="\t",
        )[0].tolist()
    else:
        raise Exception("Version not supported")
    return df


def postprocess(df, col):
    res = [re.sub("\n", "", str(x)) for x in df[col]]
    res = [x.split("                ")[0] for x in res]
    res = [x.split("Note")[0] for x in res]
    res = ["" if x[1:4] == '"""' else x for x in res]
    return res


def extract_between_markers(input_string):
    pattern = r"@@(.*?)##"
    extracted = re.findall(pattern, input_string)
    return extracted


def computeTermEvalMetrics(extracted_terms, gold_df):
    # make lower case cause gold standard is lower case
    extracted_terms = [item.lower() for item in extracted_terms]
    extracted_terms = set([x for x in extracted_terms if x != ""])
    gold_set = [item.lower() for item in gold_df]
    gold_set = set([x for x in gold_set if x != ""])
    true_pos = extracted_terms.intersection(gold_set)
    recall = round(len(true_pos) * 100 / len(gold_set), 1)
    precision = round(len(true_pos) * 100 / len(extracted_terms), 1)
    fscore = round(2 * (precision * recall) / (precision + recall), 1)
    print(str(precision) + " & " + str(recall) + " & " + str(fscore))
    return precision, recall, fscore


def eval_template3(df, col, gold_path):
    output = postprocess(df, col)
    output = [extract_between_markers(x) for x in output]
    predictions = []
    for x in output:
        predictions.extend(x)

    gold_list = pd.read_csv(gold_path, header=None, delimiter="\t")[0].tolist()
    computeTermEvalMetrics(predictions, gold_list)


def extract_entities(words, labels):
    entities = []
    current_entity = ""
    current_label = ""

    for word, label in zip(words, labels):
        if label.startswith("B"):
            if current_entity:
                entities.append(current_entity.strip())
            current_entity = word
            current_label = label[2:]
        elif label.startswith("I") and current_label == label[2:]:
            current_entity += " " + word
        else:
            if current_entity:
                entities.append(current_entity.strip())
                current_entity = ""
                current_label = ""

    if current_entity:
        entities.append(current_entity.strip())

    return entities


def filter_text(text):
    allowed_chars = {"B", "I", "O", " "}
    filtered_text = "".join(c for c in text if c in allowed_chars)
    return filtered_text


def eval_template1(df, col, gold_list):
    output = postprocess(df, col)
    output = [re.sub("'", "", x).strip() for x in output]

    extracted_list = []
    for x, y in list(zip(df.words, output)):
        words = eval(x)
        labels = y[: len(words) * 2]

        if labels.startswith("'I") or labels == "":
            labels = "O " * len(words)
        labels = filter_text(labels).split(" ")
        extracted_list.append(labels)

    predictions = []
    for x, y in zip(df.words, extracted_list):
        terms = extract_entities(eval(x), y)
        predictions.extend(terms)
    precision, recall, fscore = computeTermEvalMetrics(predictions, gold_list)
    return precision, recall, fscore


def evaluate_prompts(df, format, gold_list):
    if "1" in format:
        print("#" * 50)
        print("#1. Extracted IOB format")
        precision, recall, fscore = eval_template1(df, format, gold_list)

    elif "2" in format:
        print("#" * 50)
        print("#2. Extracted candidate term list")
        candidate_terms = []
        for x in df[format]:
            ## 13B ANN
            if (
                "[global, proteomics, pathway, analysis, pressure-overload-induced, heart, failure, mitochondrial-targeted, peptides]"
                in str(x)
            ):
                candidate_terms.extend(
                    [
                        "global",
                        "proteomics",
                        "pathway",
                        "analysis",
                        "pressure-overload-induced",
                        "heart",
                        "failure",
                        "mitochondrial-targeted",
                        "peptides",
                    ]
                )
            ## 13B NES
            elif (
                '[ "diuretics", "comorbidities", "6-MWT", "NYHA class III", "age > 65 years" ]'
                in str(x)
            ):
                candidate_terms.extend(
                    [
                        "diuretics",
                        "comorbidities",
                        "6-MWT",
                        "NYHA class III",
                        "age > 65 years",
                    ]
                )
            elif (
                '["patient-level regression", "30-day risk-adjusted mortality", "readmissions", "costs", "volume groups", "patient", "physician", "hospital characteristics"]'
                in str(x)
            ):
                candidate_terms.extend(
                    [
                        "patient-level regression",
                        "30-day risk-adjusted mortality",
                        "readmissions",
                        "costs",
                        "volume groups",
                        "patient",
                        "physician",
                        "hospital characteristics",
                    ]
                ) 
            elif  'Effecten:' in str(x) and lang == 'nl':
                candidate_terms.extend([])
            else:
                term = eval(str(x).split('Output: ')[1].split('[INST')[0].split('Please')[0].split('Note')[0].strip()) if len(str(x).split('Output: ')) > 1 and len(str(x).split('Output: ')[1].split('[INST')[0].strip()) > 1 else []
                candidate_terms.extend(term)
                )
                candidate_terms.extend(term)
        precision, recall, fscore = computeTermEvalMetrics(candidate_terms, gold_list)
    elif "3" in format:
        print("#" * 50)
        print("#3. Masking terms")
        ### RAW
        candidate_terms = []
        for x in df[format]:
            sent = (
                str(x).split("Output: ")[1].split("[INST]")[0].strip()
                if len(str(x).split("Output: ")) > 1
                else ""
            )
            candidate_terms.extend(extract_between_markers(sent))

        ### PROCESSED
        candidate_terms_list = []
        for x in candidate_terms:
            candidate_terms_list.append(x.split("@@")[-1])

        precision, recall, fscore = computeTermEvalMetrics(
            candidate_terms_list, gold_list
        )
    else:
        raise Exception("Format not supported")
    return precision, recall, fscore


if __name__ == "__main__":
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
    evaluate_prompts(df, args.format, args.lang, args.ver, gold_list)
