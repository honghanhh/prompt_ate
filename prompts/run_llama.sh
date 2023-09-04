# Only for template 2
python llama2.py  --lang en --ver ann  --formats 2 --output_path ../results/prompt_results/llama_en_htfl_preds_ann.csv
python llama2.py  --lang en --ver nes  --formats 2 --output_path ../results/prompt_results/llama_en_htfl_preds_nes.csv

python llama2.py  --lang fr --ver ann  --formats 2 --output_path ../results/prompt_results/llama_fr_htfl_preds_ann.csv
python llama2.py  --lang fr --ver nes  --formats 2 --output_path ../results/prompt_results/llama_fr_htfl_preds_nes.csv

python llama2.py  --lang nl --ver ann  --formats 2 --output_path ../results/prompt_results/llama_fr_htfl_preds_ann.csv
python llama2.py  --lang nl --ver nes  --formats 2 --output_path ../results/prompt_results/llama_fr_htfl_preds_nes.csv
