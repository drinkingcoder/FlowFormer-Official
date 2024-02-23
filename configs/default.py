from yacs.config import CfgNode as CN
_CN = CN()

_CN.name = 'default'
_CN.suffix ='arxiv2'
_CN.gamma = 0.8
_CN.max_flow = 400
_CN.batch_size = 12
_CN.sum_freq = 50
_CN.val_freq = 1500
_CN.image_size = [368, 496]
_CN.add_noise = True
_CN.critical_params = []

_CN.transformer = 'latentcostformer'
_CN.restore_ckpt = None

###########################################
# latentcostformer
_CN.latentcostformer = CN()
_CN.latentcostformer.pe = 'linear'
_CN.latentcostformer.dropout = 0.0
_CN.latentcostformer.encoder_latent_dim = 256 # in twins, this is 256
_CN.latentcostformer.query_latent_dim = 64
_CN.latentcostformer.cost_latent_input_dim = 64
_CN.latentcostformer.cost_latent_token_num = 8
_CN.latentcostformer.cost_latent_dim = 128
_CN.latentcostformer.predictor_dim = 128
_CN.latentcostformer.motion_feature_dim = 209 # use concat, so double query_latent_dim
_CN.latentcostformer.arc_type = 'transformer'
_CN.latentcostformer.cost_heads_num = 1
# encoder
_CN.latentcostformer.pretrain = True
_CN.latentcostformer.context_concat = False
_CN.latentcostformer.encoder_depth = 3
_CN.latentcostformer.feat_cross_attn = False
_CN.latentcostformer.patch_size = 8
_CN.latentcostformer.patch_embed = 'single'
_CN.latentcostformer.gma = True
_CN.latentcostformer.rm_res = True
_CN.latentcostformer.vert_c_dim = 64
_CN.latentcostformer.cost_encoder_res = True
_CN.latentcostformer.cnet = 'twins'
_CN.latentcostformer.fnet = 'twins'
_CN.latentcostformer.only_global = False
_CN.latentcostformer.add_flow_token = True
_CN.latentcostformer.use_mlp = False
_CN.latentcostformer.vertical_conv = False
# decoder
_CN.latentcostformer.decoder_depth = 12
_CN.latentcostformer.critical_params = ['cost_heads_num', 'vert_c_dim', 'cnet', 'pretrain' , 'add_flow_token', 'encoder_depth', 'gma', 'cost_encoder_res']
##########################################

### TRAINER
_CN.trainer = CN()
_CN.trainer.scheduler = 'OneCycleLR'

_CN.trainer.optimizer = 'adamw'
_CN.trainer.canonical_lr = 25e-5
_CN.trainer.adamw_decay = 1e-4
_CN.trainer.clip = 1.0
_CN.trainer.num_steps = 50000
_CN.trainer.epsilon = 1e-8
_CN.trainer.anneal_strategy = 'linear'
def get_cfg():
    return _CN.clone()
