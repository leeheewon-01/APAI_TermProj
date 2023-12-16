
python caption_mplug_scst.py \
    --config ./configs/caption_mplug_base_scst.yaml \
    --output_dir /abr/cvmi_lhw/mPLUG_output/coco_caption_base_scst \
    --checkpoint /abr/cvmi_lhw/mPLUG/mplug_base.pth \
    --text_encoder bert-base-uncased \
    --text_decoder bert-base-uncased \
    --do_two_optim \
    --min_length 8 \
    --max_length 25 \
    --max_input_length 25 \
    --do_amp
