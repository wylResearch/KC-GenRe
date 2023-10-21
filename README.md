
# First Stage
## 1. train and inference
- Wiki27K and FB15K-237-N
```

python ckg_TuckER/main.py --dataset Wiki27K --num_iterations 500 --batch_size 256 --lr 0.0005 --dr 1.0 --edim 256 --rdim 256 --input_dropout 0.3 --hidden_dropout1 0.4 --hidden_dropout2 0.5 --label_smoothing 0.1 --model tucker

python ckg_TuckER/main.py --dataset FB15K-237-N --num_iterations 500 --batch_size 256 --lr 0.0005 --dr 1.0 --edim 256 --rdim 256 --input_dropout 0.3 --hidden_dropout1 0.4 --hidden_dropout2 0.5 --label_smoothing 0.1 --model tucker
```

- ReVerb20K and ReVerb45K
```
python openkg_CEKFA_conv/main_rank.py --dataset ReVerb20K --model_name BertResNet_2Inp --bmn bert-base-uncased --bert_init --np_neigh_mth LAN --rp_neigh_mth Local --maxnum_rpneighs 5 --rerank_Top_K 30 --cuda 0

python openkg_CEKFA_conv/main_rank.py --dataset ReVerb45K --model_name BertResNet_2Inp --bmn bert-base-uncased --bert_init --np_neigh_mth LAN --rp_neigh_mth Local --maxnum_rpneighs 10 --rerank_Top_K 30 --cuda 0
```

## 2. prepare data for second stage
We will organize this portion of the code later.
- Wiki27K and FB15K-237-N: ckg_TuckER/process_rerank_data_ckg.py

- ReVerb20K and ReVerb45K：openkg_CEKFA_conv/process_rerank_data


# Second Stage：KC-GenRe
training and inference: qlora-train.py

evaluate: qlora_evaluate_rerank_ckg.py or qlora_evaluate_rerank_openkg.py

Please modify the file paths accordingly.

- FB15K-237-N
```
# training
CUDA_VISIBLE_DEVICES=0 python qlora-train.py --model_name_or_path /data/pretrained_model/llama-7b --output_dir ./output/fb-7b-codev2-data_t2_fillwen_Top20_defi1_rl01 --dataset_train ./results/FB15K-237-N/tucker_256/rerank_data_new/FB15K-237-N_rerank_data_train_rs_t2_Top20_defi1.json --dataset_eval ./results/FB15K-237-N/tucker_256/rerank_data_new/FB15K-237-N_rerank_data_valid_rs_t2_Top20_defi1.json --do_train --do_eval --source_max_len 2048 --target_max_len 2048 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --gradient_accumulation_steps 1 --save_strategy steps --data_seed 42 --seed 0 --evaluation_strategy steps --max_new_tokens 32 --dataloader_num_workers 3 --group_by_length --logging_strategy steps --lora_r 64 --lora_alpha 16 --lora_modules all --double_quant --quant_type nf4 --bf16 --bits 4 --warmup_ratio 0.03 --lr_scheduler_type constant --gradient_checkpointing --logging_steps 100 --max_steps 28050 --eval_steps 9350 --save_steps 9350 --save_total_limit 5 --learning_rate 0.0001 --adam_beta2 0.999 --max_grad_norm 0.3 --lora_dropout 0.05 --weight_decay 0.0 --report_to wandb --use_option_scores --rankloss_weight 0.1
# inference
CUDA_VISIBLE_DEVICES=0 python qlora-train.py --model_name_or_path /data/pretrained_model/llama-7b --output_dir ./output/fb-7b-codev2-data_t2_fillwen_Top20_defi1_rl01 --dataset_test ./results/FB15K-237-N/tucker_256/rerank_data_new/FB15K-237-N_rerank_data_test_rs_t2_Top20_defi1.json --do_train False --do_eval False --do_predict True --per_device_eval_batch_size 16 --source_max_len 2048 --target_max_len 2048 --predict_with_generate --max_new_tokens 200 --save_file_name Top20_greedy_csgen --use_constraint_gen --gen_TopK 20
# evaluate
python qlora_evaluate_rerank_ckg.py --data_path ./dataset/ --dataset FB15K-237-N --test_data_rank_preds_path ./results/FB15K-237-N/tucker_256/preds_Top50_test.json --test_data_rerank_inp_path ./results/FB15K-237-N/tucker_256/rerank_data_new/FB15K-237-N_rerank_data_test_rs_t2_Top20.json --test_data_rerank_gen_path  ./output/fb-7b-codev2-data_t2_fillwen_Top20_defi1_rl01/rerank_predictions_Top20_greedy_csgen.jsonl --rerank_Top_K 20
```
- Wiki27K
```
CUDA_VISIBLE_DEVICES=0 python qlora-train.py --model_name_or_path /data/pretrained_model/llama-7b --output_dir ./output/wiki-7b-codev2-data_t2_fillwen_Top20_defi1_rl01 --dataset_train ./results/Wiki27K/tucker_256/rerank_data_new/Wiki27K_rerank_data_train_rs_t2_Top20_defi1.json --dataset_eval ./results/Wiki27K/tucker_256/rerank_data_new/Wiki27K_rerank_data_valid_rs_t2_Top20_defi1.json --do_train --do_eval --source_max_len 2048 --target_max_len 2048 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --gradient_accumulation_steps 1 --save_strategy steps --data_seed 42 --seed 0 --evaluation_strategy steps --max_new_tokens 32 --dataloader_num_workers 3 --group_by_length --logging_strategy steps --lora_r 64 --lora_alpha 16 --lora_modules all --double_quant --quant_type nf4 --bf16 --bits 4 --warmup_ratio 0.03 --lr_scheduler_type constant --gradient_checkpointing --logging_steps 100 --max_steps 28050 --eval_steps 9350 --save_steps 9350 --save_total_limit 5 --learning_rate 0.0001 --adam_beta2 0.999 --max_grad_norm 0.3 --lora_dropout 0.05 --weight_decay 0.0 --report_to wandb --use_option_scores --rankloss_weight 0.1
CUDA_VISIBLE_DEVICES=0 python qlora-train.py --model_name_or_path /data/pretrained_model/llama-7b --output_dir ./output/wiki-7b-codev2-data_t2_fillwen_Top20_defi1_rl01 --dataset_test ./results/Wiki27K/tucker_256/rerank_data_new/Wiki27K_rerank_data_test_rs_t2_Top20_defi1.json --do_train False --do_eval False --do_predict True --per_device_eval_batch_size 16 --source_max_len 2048 --target_max_len 2048 --predict_with_generate --max_new_tokens 200 --save_file_name Top20_greedy_csgen --use_constraint_gen --gen_TopK 20
python qlora_evaluate_rerank_ckg.py --data_path ./dataset/ --dataset Wiki27K --test_data_rank_preds_path ./results/Wiki27K/tucker_256/preds_Top50_test.json --test_data_rerank_inp_path ./results/Wiki27K/tucker_256/rerank_data_new/Wiki27K_rerank_data_test_rs_t2_Top20.json --test_data_rerank_gen_path  ./output/wiki-7b-codev2-data_t2_fillwen_Top20_defi1_rl01/rerank_predictions_Top20_greedy_csgen.jsonl --rerank_Top_K 20

```

- ReVerb20K
```
CUDA_VISIBLE_DEVICES=0 python qlora-train.py --model_name_or_path /data/pretrained_model/llama-7b --output_dir ./output/ReVerb20K-7b-codev2-data_t2_fillwen_Top30_rl03 --dataset_train ./results/ReVerb20K/BertResNet_2Inp_1.0_768_0.0001_npLAN_rpLocal_Num5_bertinit_2023-05-30/rerank_data_new/ReVerb20K_rerank_data_train_rs_t2_fillwen_Top30.json --dataset_eval ./results/ReVerb20K/BertResNet_2Inp_1.0_768_0.0001_npLAN_rpLocal_Num5_bertinit_2023-05-30/rerank_data_new/ReVerb20K_rerank_data_valid_rs_t2_fillwen_Top30.json --do_train --do_eval --source_max_len 2048 --target_max_len 2048 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --gradient_accumulation_steps 1 --save_strategy steps --data_seed 42 --seed 0 --evaluation_strategy steps --max_new_tokens 32 --dataloader_num_workers 3 --group_by_length --logging_strategy steps --lora_r 64 --lora_alpha 16 --lora_modules all --double_quant --quant_type nf4 --bf16 --bits 4 --warmup_ratio 0.03 --lr_scheduler_type constant --gradient_checkpointing --logging_steps 100 --max_steps 5814 --eval_steps 1938 --save_steps 1938 --save_total_limit 5 --learning_rate 0.0001 --adam_beta2 0.999 --max_grad_norm 0.3 --lora_dropout 0.05 --weight_decay 0.0 --report_to wandb --use_option_scores --rankloss_weight 0.3
CUDA_VISIBLE_DEVICES=0 python qlora-train.py --model_name_or_path /data/pretrained_model/llama-7b --output_dir ./output/ReVerb20K-7b-codev2-data_t2_fillwen_Top30_rl03 --dataset_test ./results/ReVerb20K/BertResNet_2Inp_1.0_768_0.0001_npLAN_rpLocal_Num5_bertinit_2023-05-30/rerank_data_new/ReVerb20K_rerank_data_test_rs_t2_fillwen_cand_a7_Top3_Top30_kf3.json --do_train False --do_eval False --do_predict True --per_device_eval_batch_size 16 --source_max_len 2048 --target_max_len 2048 --predict_with_generate --max_new_tokens 200 --save_file_name Top30_greedy_kf3_cand_a7_Top3_csgen --use_constraint_gen --gen_TopK 30
python qlora_evaluate_rerank_openkg.py --data_path ./dataset/ --dataset ReVerb20K --test_data_rank_preds_path ./results/ReVerb20K/BertResNet_2Inp_1.0_768_0.0001_npLAN_rpLocal_Num5_bertinit_2023-05-30/preds_Top100_test.json --test_data_rerank_inp_path ./results/ReVerb20K/BertResNet_2Inp_1.0_768_0.0001_npLAN_rpLocal_Num5_bertinit_2023-05-30/rerank_data_new/ReVerb20K_rerank_data_test_rs_t2_fillwen_Top30.json --test_data_rerank_gen_path ./output/ReVerb20K-7b-codev2-data_t2_fillwen_Top30_rl03/rerank_predictions_Top30_greedy_kf3_cand_a7_Top3_csgen.jsonl --rerank_Top_K 30

```

- ReVerb45K
```
CUDA_VISIBLE_DEVICES=0 python qlora-train.py --model_name_or_path /data/pretrained_model/llama-7b --output_dir ./output/ReVerb45K-7b-codev2-data_t2_fillwen_Top30_rl1 --dataset_train ./results/ReVerb45K/BertResNet_2Inp_1.0_768_0.0001_npLAN_rpLocal_Num10_bertinit_2023-05-30/rerank_data_new/ReVerb45K_rerank_data_train_rs_t2_fillwen_Top30.json --dataset_eval ./results/ReVerb45K/BertResNet_2Inp_1.0_768_0.0001_npLAN_rpLocal_Num10_bertinit_2023-05-30/rerank_data_new/ReVerb45K_rerank_data_valid_rs_t2_fillwen_Top30.json --do_train --do_eval --source_max_len 2048 --target_max_len 2048 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --gradient_accumulation_steps 1 --save_strategy steps --data_seed 42 --seed 0 --evaluation_strategy steps --max_new_tokens 32 --dataloader_num_workers 3 --group_by_length --logging_strategy steps --lora_r 64 --lora_alpha 16 --lora_modules all --double_quant --quant_type nf4 --bf16 --bits 4 --warmup_ratio 0.03 --lr_scheduler_type constant --gradient_checkpointing --logging_steps 100 --max_steps 5814 --eval_steps 1938 --save_steps 1938 --save_total_limit 5 --learning_rate 0.0001 --adam_beta2 0.999 --max_grad_norm 0.3 --lora_dropout 0.05 --weight_decay 0.0 --report_to wandb --use_option_scores --rankloss_weight 1.0
CUDA_VISIBLE_DEVICES=0 python qlora-train.py --model_name_or_path /data/pretrained_model/llama-7b --output_dir ./output/ReVerb45K-7b-codev2-data_t2_fillwen_Top30_rl1 --dataset_test ./results/ReVerb45K/BertResNet_2Inp_1.0_768_0.0001_npLAN_rpLocal_Num10_bertinit_2023-05-30/rerank_data_new/ReVerb45K_rerank_data_test_rs_t2_fillwen_cand_a7_Top3_Top30_kf3.json --do_train False --do_eval False --do_predict True --per_device_eval_batch_size 16 --source_max_len 2048 --target_max_len 2048 --predict_with_generate --max_new_tokens 200 --save_file_name Top30_greedy_kf3_cand_a7_Top3_csgen --use_constraint_gen --gen_TopK 30
python qlora_evaluate_rerank_openkg.py --data_path ./dataset/ --dataset ReVerb45K --test_data_rank_preds_path ./results/ReVerb45K/BertResNet_2Inp_1.0_768_0.0001_npLAN_rpLocal_Num10_bertinit_2023-05-30/preds_Top100_test.json --test_data_rerank_inp_path ./results/ReVerb45K/BertResNet_2Inp_1.0_768_0.0001_npLAN_rpLocal_Num10_bertinit_2023-05-30/rerank_data_new/ReVerb45K_rerank_data_test_rs_t2_fillwen_Top30.json --test_data_rerank_gen_path ./output/ReVerb45K-7b-codev2-data_t2_fillwen_Top30_rl1/rerank_predictions_Top30_greedy_kf3_cand_a7_Top3_csgen.jsonl --rerank_Top_K 30

```