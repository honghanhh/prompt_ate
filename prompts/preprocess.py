import pandas as pd
import argparse
import glob

def split_list(lst, sep):
    return [i.split() for i in ' '.join(lst).split(sep)]

def convert_data(path):
    df = pd.read_csv(path, skip_blank_lines=False,
                     delimiter = '\t', header=None, names = ['words','labels'])
    df = df.fillna('This is an end of a sentence.')
    df = pd.DataFrame({'words': split_list(df['words'].tolist(), 'This is an end of a sentence.'),
                   'labels': split_list(df['labels'].tolist(), 'This is an end of a sentence.')})
    return df[:-1]

def read_data(input_path):
    files = sorted(glob.glob(input_path + '*.tsv'))
    df = []
    for f in files:
        # print(f.split('/')[-1])
        temp = convert_data(f)
        df.append(temp)
    df = pd.concat(df)
    df['text'] = [' '.join(x) for x in df['words']]
    return df

def merge_corpus(input_path1, input_path2):
    ann = read_data(input_path1)
    nes = read_data(input_path2)
    df = pd.DataFrame({'words': ann['words'],
                       'text': ann['text'],
                       'ann': ann['labels'],
                       'nes': nes['labels']
                     })
    df['difference'] = [0 if x==y else 1 for  x,y in zip(df['ann'], df['nes'])]
    return df
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data_path", type=str, required=True)
    parser.add_argument("--data_path1", default="data_path1", type=str, required=True)
    parser.add_argument("--output_path", default="output_path", type=str, required=True)
    args = parser.parse_args()

    df = merge_corpus(args.data_path, args.data_path1)
    df.to_csv(args.output_path, index=False)
