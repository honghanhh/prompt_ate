python evaluate.py --data_path ../ACTER/processed_data/en_htfl.csv --lang en --ver ann --output_path en_htfl_preds_ann.csv
python evaluate.py --data_path ../ACTER/processed_data/en_htfl.csv --lang en --ver nes --output_path en_htfl_preds_nes.csv

python evaluate.py --data_path ../ACTER/processed_data/fr_htfl.csv --lang fr --ver ann --output_path fr_htfl_preds_ann.csv
python evaluate.py --data_path ../ACTER/processed_data/fr_htfl.csv --lang fr --ver nes --output_path fr_htfl_preds_nes.csv

python evaluate.py --data_path ../ACTER/processed_data/nl_htfl.csv --lang nl --ver ann --output_path nl_htfl_preds_ann.csv
python evaluate.py --data_path ../ACTER/processed_data/nl_htfl.csv --lang nl --ver nes --output_path nl_htfl_preds_nes.csv
