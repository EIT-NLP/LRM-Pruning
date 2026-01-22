num_gpus=2
accelerate launch --num-processes $num_gpus --multi-gpu --num-machines 1 -m eval.eval \
    --task MATH500 \
    --batch_size 16 \
    --model_args "pretrained=../SKIPGPT/models/Llama31-8B-instruct-fullsft-openthoughts/" \
    --output_path cot_eval/slicegpt_0.2 

accelerate launch --num-processes $num_gpus --multi-gpu --num-machines 1 -m eval.eval \
    --task JEEBench \
    --batch_size 16 \
    --model_args "pretrained=../SKIPGPT/models/Llama31-8B-instruct-fullsft-openthoughts/" \
    --output_path cot_eval/slicegpt_0.2 

accelerate launch --num-processes $num_gpus --multi-gpu --num-machines 1 -m eval.eval \
    --task LiveCodeBenchv5 \
    --batch_size 16 \
    --model_args "pretrained=../SKIPGPT/models/Llama31-8B-instruct-fullsft-openthoughts/" \
    --output_path cot_eval/slicegpt_0.2 
    
accelerate launch --num-processes 2 --multi-gpu --num-machines 1 -m eval.eval \
    --model hf \
    --task JEEBench \
    --batch_size 16 \
    --max_tokens 16384 \
    --model_args "pretrained=../SKIPGPT/models/Llama31-8B-instruct-fullsft-openthoughts/,slice_model_path=/code/CoT_Baseline/TransformerCompression/slicegpt_results/slice_model/slice_0.2/,adapter_path=/code/CoT_Baseline/TransformerCompression/slicegpt_results/lora/lora_0.2_ckpt/checkpoint-1000/" \
    --output_path cot_eval/JEEBench/slicegpt_0.2 \
    --debug 