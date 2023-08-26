import os
os.environ["CUDA_VISIBLE_DEVICES"]='7'
import torch  
torch.manual_seed(3407)
import random
random.seed(3407)
import numpy as np
np.random.seed(3407)

import pandas as pd
from seq2seq_model import Seq2SeqModel

import logging
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore")
import timeit

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Just an example",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-lang", "--lang", type=str, help='en')
    parser.add_argument("-version", "--version", type=str, help='ann')
    parser.add_argument("-output_dir", "--output_dir",type=str, help="./exp/en")
    
    args = parser.parse_args()
        
    start = timeit.default_timer()

    print("Loading dataset...")
    path = './template_datasets/'
    train1 = pd.read_csv(path + args.lang + '_htfl_'+ args.version + '.csv')
    train2 = pd.read_csv(path + args.lang + '_corp_'+ args.version + '.csv')
    train_df = pd.concat([train1, train2], ignore_index=True)
    eval_df = pd.read_csv(path + args.lang + '_wind_'+ args.version + '.csv')
    train_df['input_text'] = [x.lower() for x in train_df['input_text']]
    train_df['target_text'] = [x.lower() for x in train_df['target_text']]
    eval_df['input_text'] = [x.lower() for x in eval_df['input_text']]
    eval_df['target_text'] = [x.lower() for x in eval_df['target_text']]

    print("Loading model...")
    model_args = {
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "max_seq_length": 70,
        "train_batch_size": 32,
        "num_train_epochs": 5,
        "save_eval_checkpoints": False,
        "save_model_every_epoch": False,
        "evaluate_during_training": True,
        "evaluate_generated_text": True,
        "evaluate_during_training_verbose": True,
        "use_multiprocessing": False,
        "max_length": 35,
        "manual_seed": 4,
        "save_steps": 11898,
        "gradient_accumulation_steps": 1,
        "output_dir": args.output_dir
    }

    # Initialize model
    model = Seq2SeqModel(
        encoder_decoder_type= "mbart",
        encoder_decoder_name="facebook/mbart-large-50-many-to-many-mmt",
        args=model_args,
    )
    print("Training model...")
    # Train the model
    model.train_model(train_df, eval_data=eval_df)

    results = model.eval_model(eval_df, 
                               verbose = False, 
                               silent = True)
    print(results)

    print("Finishing model...")

    stop = timeit.default_timer()

    print('Time: ', stop - start) 