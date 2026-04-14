python eval_bertscore.py \
    --input-csv path/to/generated_captions.csv \
    --model-type roberta-large \
    --batch-size 64 \
    --save-per-sample \
    --output-dir path/to/output \
    --output-name two_stage_coca_L-14_bertscore
