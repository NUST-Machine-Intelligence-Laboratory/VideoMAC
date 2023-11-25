CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 python main_pretrain.py \
--model mac_r18 \
--patch_size 32 \
--batch_size 512 --update_freq 1 \
--blr 1e-3 \
--mask_ratio 0.75 \
--gamma 1.0 \
--momentum_target 0.996 \
--use_amp False \
--epochs 100 \
--warmup_epochs 10 \
--data_path ./data/ytb18 \
--output_dir ./checkpoints/mac-r18-r0.75

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 python main_pretrain.py \
--model mac_r50 \
--patch_size 32 \
--batch_size 512 --update_freq 1 \
--blr 1e-3 \
--mask_ratio 0.75 \
--gamma 1.0 \
--momentum_target 0.996 \
--use_amp False \
--epochs 100 \
--warmup_epochs 10 \
--data_path ./data/ytb18 \
--output_dir ./checkpoints/mac-r50-r0.75
