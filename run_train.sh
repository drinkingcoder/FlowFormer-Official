mkdir -p checkpoints

NUM_GPUS=6
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

python -u train_FlowFormer.py --name chairs --stage chairs --validation chairs --num_gpus=${NUM_GPUS}
python -u train_FlowFormer.py --name things --stage things --validation sintel --num_gpus=${NUM_GPUS}
python -u train_FlowFormer.py --name sintel --stage sintel --validation sintel --num_gpus=${NUM_GPUS}
python -u train_FlowFormer.py --name kitti --stage kitti --validation kitti --num_gpus=${NUM_GPUS}