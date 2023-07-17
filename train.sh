echo "Conducting training..."
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
GPU_NUM=2
BATCH=16

CFG=yolox_relu_pretrain.py
python -m yolox.tools.train -f ${CFG} -d ${GPU_NUM} -b ${BATCH} -o --fp16 --logger wandb wandb-project "YOLOX-pretrain"
