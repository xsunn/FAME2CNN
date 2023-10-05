import sys
sys.path.append('')
import os
os.environ['CUDA_VISIBLE_DEVICES']=','.join(map(str,[0]))

import torchmetrics as metrics
import torchmetrics.classification as clf_metrics
import pickle

from cardio.pylightning.custom_kfold import kfold_fit
import cardio.resolvers as resolvers
import configurationGIN as cfg

import cardio.utils as utils
from cardio.hypertune import hypertune
import torch


def do_simple_run(confs):
    ''' Does a simple run:
        (1) Train and validate
        (2) Test with the best model (if stored)
    '''
    logger = utils.TrainLogger(confs)
    val_metrics = [
        clf_metrics.BinaryAccuracy(task='binary'), clf_metrics.BinaryPrecision(task='binary'),
        clf_metrics.BinaryRecall(task='binary'), clf_metrics.BinaryF1Score(task='binary'), metrics.AUROC(task='binary',pos_label=1)
    ]
    # Resolvers
    val_metrics, test_metrics = resolvers.metrics_resolver(confs)
    datamodule = resolvers.datamodule_resolver(confs)
    datamodule.setup(None) # hacky but we need it before we call model.resolver
    trainer = resolvers.trainer_resolver(confs)
    model = resolvers.model_resolver(confs, logger, val_metrics, test_metrics)

    # Train - Test
    trainer.fit(model, datamodule=datamodule)
    # trainer.test(ckpt_path="best", datamodule=datamodule)
    # trainer.test(ckpt_path="//home/sun/project/FAME2Pre/checkpoint/K5S-fold1-6-128-mean/epoch=218-step=3942-val_f1=0.63.ckpt",datamodule=datamodule)
    trainer.test(ckpt_path="/home/sun/project/FAME2CNN/checkpoint/resnet-fold1/epoch=12-step=546-val_f1=0.46.ckpt",datamodule=datamodule)
    # trainer.test(ckpt_path="best", datamodule=datamodule)


def main():
    confs = cfg.parse_args()
    # print(confs)

    # confs.train_batch_size=30
    # confs.test_batch_size=30
    # confs.val_batch_size=30
# gnn_nb_layers : 5      gnn_dropout : 0.5336753918531467      gnn_node_emb_dim : 1024      node_pooling : mean      learning_rate : 0.0030650668140405324      weight_decay : 4.930451191799806e-05      adam_beta1 : 0.6590993957842993      use_balanced_batches : True      
# gnn_nb_layers : 8      gnn_dropout : 0.5495037441401533      gnn_node_emb_dim : 512      node_pooling : mean      learning_rate : 0.008028533446871877      weight_decay : 6.541460589500209e-05      adam_beta1 : 0.7736293642471647      use_balanced_batches : False     
# gnn_nb_layers : 3      gnn_dropout : 0.5597603205849672      gnn_node_emb_dim : 512      node_pooling : sum      learning_rate : 0.0018126739567448577      weight_decay : 9.136928972034532e-05      adam_beta1 : 0.5167723777290408      use_balanced_batches : False      
# gnn_nb_layers : 7      gnn_dropout : 0.31925440919895615      gnn_node_emb_dim : 512      node_pooling : all_stats      learning_rate : 0.0007422730361870196      weight_decay : 7.145714718246155e-05      adam_beta1 : 0.7703603551510814      use_balanced_batches : False      
# gnn_nb_layers : 5      gnn_dropout : 0.37336442308819295      gnn_node_emb_dim : 512      node_pooling : mean      learning_rate : 0.0038614954379376774      weight_decay : 5.329687557709598e-05      adam_beta1 : 0.703576033383889      use_balanced_batches : False      
# k10 gnn_nb_layers : 8      gnn_dropout : 0.3172682083338288      gnn_node_emb_dim : 64      node_pooling : mean      learning_rate : 0.0016433241963198307      weight_decay : 7.594769970503184e-05      adam_beta1 : 0.7586869860337067      use_balanced_batches : True      
# gnn_nb_layers : 7      gnn_dropout : 0.2574135488746098      gnn_node_emb_dim : 1024      node_pooling : all_stats      learning_rate : 0.000655302514628885      weight_decay : 1.167323876120324e-06      adam_beta1 : 0.6346867290760816      use_balanced_batches : True      
# d5s gnn_nb_layers : 10      gnn_dropout : 0.5931476865499534      gnn_node_emb_dim : 1024      node_pooling : all_stats      learning_rate : 0.0005035622678224119      weight_decay : 6.679017334462597e-05      adam_beta1 : 0.601765275614863      use_balanced_batches : True      
# K5S gnn_nb_layers : 7      gnn_dropout : 0.5694367117766881      gnn_node_emb_dim : 512      node_pooling : all_stats      learning_rate : 0.004505040547581704      weight_decay : 4.5538260055129215e-05      adam_beta1 : 0.5237497113058881      use_balanced_batches : True      
    confs.run_mode = "simple"
    confs.dataset_dir ="/home/sun/data/cnn/patch/"
    confs.task = "baseline"
    confs.gnn_nb_layers=6
    confs.gnn_dropout=0.35694367117766881
    confs.gnn_node_emb_dim=256
    confs.node_pooling='mean'
    confs.learning_rate=0.001505040547581704
    confs.weight_decay=4.5538260055129215e-05 
    confs.adam_beta1=0.5237497113058881
    confs.use_balanced_batches=True
    confs.gnn_is_siamese=True
    confs.nb_epochs=1
    confs.save_hyper_path="/home/sun/project/FAME2Pre/hyperginfold5.txt"
    confs.loss_func='CrossEntropyLoss'   
    confs.gnn_pooling_cls_name='ConfigurablePooling'  
    print(confs.dataset_dir,confs.gnn_is_siamese)
    do_simple_run(confs)

    # confs.device=torch.device('cuda:1')
    # do_simple_run(confs)
    # if confs.run_mode == "hypertune":
#     do_hypertune(confs)
    # elif confs.run_mode == "simple":
    #     do_simple_run(confs)
    # elif confs.run_mode == "kfold":
    #     do_kfold_run(confs)
    # elif confs.run_mode == "lol":
    #     pre_train(confs)
    # else:
    #     raise RuntimeError(f"run mode '{confs.run_mode}' not found!")

if __name__ == "__main__":
    main()
