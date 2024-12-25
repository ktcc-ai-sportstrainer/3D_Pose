set -x

CONFIG="AlphaPose/configs/halpe_coco_wholebody_136/resnet/256x192_res50_lr1e-3_2x-regression.yaml"
CKPT="AlphaPose/pretrained_models/multi_domain_fast50_regression_256x192.pth"
VIDEO="sample.mp4"
OUTDIR=${4:-"output"}

python inference.py \
    --cfg ${CONFIG} \
    --checkpoint ${CKPT} \
    --video ${VIDEO} \
    --outdir ${OUTDIR} \
    --detector yolo  --save_img --save_video
