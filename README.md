# Is Prompting What Term Extraction Needs?

## 1. Description

In this repo, we implement our research on the applicability of large-scale language models (LLMs) on ATE tasks in three forms of prompting: (1) sequence-labeling response; (2) text-generative response; and (3) filling the gap of both types. We conduct experiments on ACTER corpora of three languages and four domains. Check out our paper at TSD Conference: [here](https://link.springer.com/chapter/10.1007/978-3-031-70563-2_2)

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

Download the ACTER dataset at [here](https://github.com/AylaRT/ACTER) and save into ACTER folder.

## 4. Implementation

### 4.1 Sequence-labeling XLMR baseline

Please refer to the work from [ate-2022](https://github.com/honghanhh/ate-2022) for the implementation of the sequence-labeling baseline.

### 4.2 Template ATE Seq2seq ranking

Run the following command to generate the templates:

```python
cd template_ate/
python gen_template.py
```

Run the following command to train all the models:

```python
cd template_ate/
chmod +x run.sh
./run.sh
```

### 4.3. GPT-ATE Prompting

Add your API key to ```prompts/prompt_classifier.py``` and run the following command.

```python
cd prompts/
python prompt_classifier.py [--data_path] [--lang] [--ver] [--formats] [--output_path]
```

where:

- `--data_path` is the path to the data directory;
- `--lang` is the language of the corpus;
- `--ver` is the version of corpus (ANN or NES);
- `--formats` is the prompting designed format;
- `--output_path` is the path to the output csv file.

Run the following command to run all the models:

```python
cd prompts/
chmod +x run_prompt.sh
./run_prompt.sh
```

For evaluation, run the following command:

```python
cd prompts/
python evaluate.py [--data_path] [--lang] [--ver]
```

where:

- `--data_path` is the path to the data directory;
- `--lang` is the language of the corpus;
- `--ver` is the version of corpus (ANN or NES).

Run the following command to run all the evaluation:

```python
cd prompts/
chmod +x run_eval.sh
./run_eval.sh
```

### 4.4. Llama2 Prompting

Login `huggingface-clo` by Huggingface account tokens via this command

```huggingface-cli login```

and run the following command to run the model:

```python
cd prompts/
python llama2.py [--lang] [--ver] [--formats] [--output_path]
```

where:

- `--lang` is the language of the corpus;
- `--ver` is the version of corpus (ANN or NES);
- `--formats` is the prompting designed format (1,2, or 3);
- `--output_path` is the path to the output csv file.

Run the following command to run all the models:

```python
cd prompts/
chmod +x run_llama.sh
./run_llama.sh
```

## 5. Results

### 5.1. ANN gold standard

| Settings                            | English Precision | English Recall | English F1-score | French Precision | French Recall | French F1-score | Dutch Precision | Dutch Recall | Dutch F1-score |
|-------------------------------------|------------------|----------------|------------------|------------------|--------------|-----------------|----------------|--------------|----------------|
| _BIO classifier_                      |                  |                |                  |                  |              |                 |                |              |                |
| TRAIN: Wind, Equi - VAL: Corp       | 58.6             | 40.7           | 48.0             | 68.8             | 34.2         | 45.7            | 73.5           | 54.1         | 62.3           |
| TRAIN: Corp, Equi - VAL: Wind       | 58.5             | 49.5           | 53.6             | 70.7             | 41.0         | 51.9            | 73.3           | 59.7         | 65.8           |
| TRAIN: Corp, Wind - VAL: Equi       | 58.1             | 48.1           | 52.6             | 70.5             | 44.4         | 54.5            | 70.3           | 62.2         | 66.0           |
| _TemplateATE_                         |                  |                |                  |                  |              |                 |                |              |                |
| TRAIN: Wind, Equi - VAL: Corp       | 30.5             | 24.8           | 27.4             | 40.4             | 26.1         | 31.7            | 32.2           | 45.6         | 37.8           |
| TRAIN: Corp, Equi - VAL: Wind       | 24.4             | 21.3           | 22.8             | 31.7             | 26.6         | 28.9            | 29.6           | 37.4         | 33.0           |
| TRAIN: Corp, Wind - VAL: Equi       | 32.5             | 29.2           | 30.7             | 26.9             | 37.0         | 31.2            | 32.7           | 43.9         | 37.4           |
| _GPT-ATE_                            |                  |                |                  |                  |              |                 |                |              |                |
| In-domain Few-shot format #1         | 10.8             | 14.4           | 12.3             | 11.3             | 11.6         | 11.4            | 18.3           | 14.1         | 15.9           |
| In-domain Few-shot format #2         | 26.6             | 67.6           | 38.2             | 28.5             | 67.0         | 40.0            | 36.8           | 79.6         | 50.3           |
| In-domain Few-shot format #3         | 39.6             | 48.3           | 43.5             | 45.5             | 50.8         | 48.0            | 61.1           | 56.6         | 58.8           |

### 5.2. NES gold standard

| Settings                            | English Precision | English Recall | English F1-score | French Precision | French Recall | French F1-score | Dutch Precision | Dutch Recall | Dutch F1-score |
|-------------------------------------|-------------------|----------------|------------------|------------------|--------------|-----------------|----------------|--------------|----------------|
| _BIO classifier_                      |                   |                |                  |                  |              |                 |                |              |                |
| TRAIN: Wind, Equi - VAL: Corp       | 63.0              | 45.0           | 52.5             | 69.4             | 40.4         | 51.1            | 72.9           | 58.8         | 65.1           |
| TRAIN: Corp, Equi - VAL: Wind       | **63.9**          | 50.3           | **56.3**         | 72.0             | 47.2         | 57.0            | **75.9**       | 58.6         | **66.1**       |
| TRAIN: Corp, Wind - VAL: Equi       | 62.1              | **52.1**       | **56.7**         | **72.4**         | **48.5**     | **58.1**         | 73.3           | **61.5**     | **66.9**       |
| _TemplateATE_                         |                   |                |                  |                  |              |                 |                |              |                |
| TRAIN: Wind, Equi - VAL: Corp       | 30.4              | 31.5           | 31.0             | 36.4             | **39.3**     | **37.8**         | 30.4           | 45.2         | 36.4           |
| TRAIN: Corp, Equi - VAL: Wind       | 27.1              | 29.6           | 28.3             | 31.1             | 24.2         | 27.2            | **41.1**       | 37.8         | **39.4**       |
| TRAIN: Corp, Wind - VAL: Equi       | **34.7**          | **32.5**       | **33.6**         | **40.7**         | 33.0         | 36.5            | 32.2           | **47.3**     | 38.3           |
| _GPT-ATE_                             |                   |                |                  |                  |              |                 |                |              |                |
| In-domain Few-shot format #1        | 10.3              | 13.1           | 11.5             | 10.8             | 12.0         | 11.4            | 14.8           | 13.2         | 14.0           |
| In-domain Few-shot format #2        | 29.2              | **69.2**       | 41.1             | 27.9             | **66.8**     | 39.4            | 39.8           | **78.5**     | **52.8**       |
| In-domain Few-shot format #3        | **39.8**          | 53.1           | **45.5**         | **44.7**         | 54.4         | **49.1**         | **63.6**       | 60.6         | **62.1**       |


## Contributors:

- 🐮 [TRAN Thi Hong Hanh](https://github.com/honghanhh) 🐮

## License
- ___License___: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) (https://creativecommons.org/licenses/by-nc-sa/4.0/)
- ___Reference___: Please cite the following paper if you use this method for your research
```
@inproceedings{tran2024prompting,
  title={Is Prompting What Term Extraction Needs?},
  author={Tran, Hanh Thi Hong and González-Gallardo, Carlos-Emiliano and Delauney, Julien and Moreno, Jose and Doucet, Antoine and Pollak, Senja},
  booktitle={27th International Conference on Text, Speech and Dialogue (TSD 2024)},
  year={2024},
  note={Accepted}
}
```
