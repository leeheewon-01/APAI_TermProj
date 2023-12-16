NUM_GPU=1
GPU_IDS="0"
export OMP_NUM_THREADS=16
set CUDA_LAUNCH_BLOCKING=1
CUDA_VISIBLE_DEVICES=$GPU_IDS \

for lr in 1e-5
do
    python caption_mplug.py \
    --config ./configs/caption_mplug_base.yaml \
    --output_dir  /raid/cvmi_lhw/mPLUG_output/coco_caption_base_ASAM_modify_$lr \
    --checkpoint ./mplug_base.pth \
    --text_encoder bert-base-uncased \
    --text_decoder bert-base-uncased \
    --do_two_optim \
    --lr $lr \
    --min_length 8 \
    --max_length 25 \
    --max_input_length 25 \
    --accum_steps 4 \
    --do_amp
done