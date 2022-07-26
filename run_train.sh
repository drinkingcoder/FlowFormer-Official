mkdir -p checkpoints
python -u train_FlowFormer.py --name chairs --stage chairs --validation chairs
python -u train_FlowFormer.py --name things --stage things --validation sintel
python -u train_FlowFormer.py --name sintel --stage sintel --validation sintel
python -u train_FlowFormer.py --name kitti --stage kitti --validation kitti