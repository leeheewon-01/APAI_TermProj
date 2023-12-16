NUM_GPU=1
GPU_IDS="0"
export OMP_NUM_THREADS=16
set CUDA_LAUNCH_BLOCKING=1
CUDA_VISIBLE_DEVICES=$GPU_IDS \

for lr in 1e-5
do
    torchrun --nproc_per_node $NUM_GPU caption_mplug_scst.py \
    --config ./configs/caption_mplug_large_scst.yaml \
    --output_dir output/coco_caption_large_scst \
    --checkpoint mplug_large_v2.pth \
    --text_encoder bert-base-uncased \
    --text_decoder bert-base-uncased \
    --do_two_optim \
    --min_length 8 \
    --max_length 25 \
    --max_input_length 25 \
    --accum_steps 16 \
    # --do_amp
done