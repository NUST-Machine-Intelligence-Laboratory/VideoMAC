CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 python main_pretrain.py \
--model mac_cnxv1_tiny \
--patch_size 32 \
--batch_size 256 --update_freq 1 \
--blr 1e-3 \
--mask_ratio 0.75 \
--gamma 1.0 \
--momentum_target 0.996 \
--use_amp True \
--epochs 100 \
--warmup_epochs 10 \
--data_path ./data/ytb18 \
--output_dir ./checkpoints/mac-cnxv1-tiny-r0.75

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 python main_pretrain.py \
--model mac_cnxv1_small \
--patch_size 32 \
--batch_size 256 --update_freq 1 \
--blr 1e-3 \
--mask_ratio 0.75 \
--gamma 1.0 \
--momentum_target 0.996 \
--use_amp True \
--epochs 100 \
--warmup_epochs 10 \
--data_path ./data/ytb18 \
--output_dir ./checkpoints/mac-cnxv1-small-r0.75
