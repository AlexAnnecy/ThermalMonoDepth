DATA_ROOT=/media/alex/Kingston/phd/data
TRAIN_SET=$DATA_ROOT/KAIST_VIVID_256
GPU_ID=0

CUDA_VISIBLE_DEVICES=${GPU_ID} \
python train.py $TRAIN_SET \
--resnet-layers 18 \
--num-scales 1 \
--scene_type outdoor \
-b 4 \
--sequence-length 3 \
--with-ssim 1 \
--with-auto-mask 1 \
--with-pretrain 1 \
--rearrange-bin 30 \
--clahe-clip 2.0 \
--batch-size 8 \
--log-output \
--name T_vivid_resnet18_outdoor
