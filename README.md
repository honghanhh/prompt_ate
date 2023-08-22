# Is Prompting What Term Extraction Needs?

## 1. Description

In this repo, we implement our research on the applicability of large-scale language models (LLMs) on ATE tasks in three forms of prompting: (1) sequence-labeling response; (2) text-generative response; and (3) filling the gap of both types. We conduct experiments on ACTER corpora of three languages and four domains.

---

## 2. Requirements

Please install all the necessary libraries noted in [requirements.txt](./requirements.txt) using this command:

```
pip install -r requirements.txt
```

## 3. Data

The experiments were conducted on ACTER datasets:

||ACTER dataset|
|:-:|:-:|
|Languages|English, French, and Dutch|
|Domains|Corruption,  Wind energy, Equitation, Heart failure|

## 4. Implementation

### 4.1. Models

```python
cd prompts
python prompt_classifier.py [--data_path] [--lang] [--ver] [--formats] [--output_path]
```

where:

- `--data_path` is the path to the data directory;
- `--lang` is the language of the corpus;
- `--ver` is the version of corpus (ANN or NES);
- `--formats` is the prompting designed format;
- `--output_path` is the path to the output csv file.

### 4.2 Evaluation

```python
cd prompts
python evaluate.py [--data_path] [--lang] [--ver][--output_path]
```

where:

- `--data_path` is the path to the data directory;
- `--lang` is the language of the corpus;
- `--ver` is the version of corpus (ANN or NES).

## 5. Results

(updating)

## References

Tran, Hanh Thi Hong, et al. "[Can Cross-Domain Term Extraction Benefit from Cross-lingual Transfer?](https://link.springer.com/chapter/10.1007/978-3-031-18840-4_26)." Discovery Science: 25th International Conference, DS 2022, Montpellier, France, October 10–12, 2022, Proceedings. Cham: Springer Nature Switzerland, 2022.

## Contributors:
- 🐮 [TRAN Thi Hong Hanh](https://github.com/honghanhh) 🐮
