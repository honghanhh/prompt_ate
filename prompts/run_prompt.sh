python evaluate.py --data_path ../data/processed_data/en_htfl.csv --lang en --ver ann --output_path ../results/prompt_results/en_htfl_preds_ann.csv
python evaluate.py --data_path ../data/processed_data/en_htfl.csv --lang en --ver nes --output_path ../results/prompt_results/en_htfl_preds_nes.csv

python evaluate.py --data_path ../data/processed_data/fr_htfl.csv --lang fr --ver ann --output_path ../results/prompt_results/fr_htfl_preds_ann.csv
python evaluate.py --data_path ../data/processed_data/fr_htfl.csv --lang fr --ver nes --output_path ../results/prompt_results/fr_htfl_preds_nes.csv

python evaluate.py --data_path ../data/processed_data/nl_htfl.csv --lang nl --ver ann --output_path ../results/prompt_results/nl_htfl_preds_ann.csv
python evaluate.py --data_path ../data/processed_data/nl_htfl.csv --lang nl --ver nes --output_path ../results/prompt_results/nl_htfl_preds_nes.csv
