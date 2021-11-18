import torch.optim as optim
import os
from ltr.dataset import Lasot, Got10k, TrackingNet
from extern.stark.train.data import processing, sampler, LTRLoader, opencv_loader
import torch

from extern.stark.utils.box_ops import giou_loss
import importlib
import ltr.models.loss as ltr_losses
from ltr import actors
from torch.nn.functional import l1_loss
from pysot.tracker.stark_staf_tracker import STARK_ST
from extern.stark.models.stark.stark_staf_model import build_starkst
from ltr.trainers import LTRTrainer
import ltr.models.loss as ltr_losses
from ltr.models.kys.utils import DiMPScoreJittering
import extern.stark.train.data.transforms as tfm
import ltr.admin.loading as network_loading

def names2datasets(name_list: list, settings, image_loader):
    assert isinstance(name_list, list)
    datasets = []
    print(name_list)
    
    for name in name_list:
        assert name in ["LASOT", "GOT10K_vottrain", "GOT10K_votval", "GOT10K_train_full", "COCO17", "VID", "TRACKINGNET"]
        if name == "LASOT":
            if settings.use_lmdb:
                print("Building lasot dataset from lmdb")
                datasets.append(Lasot_lmdb(settings.env.lasot_lmdb_dir, split='train', image_loader=image_loader))
            else:
                datasets.append(Lasot(settings.env.lasot_dir, split='train', image_loader=image_loader))
        if name == "GOT10K_vottrain":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='vottrain', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='vottrain', image_loader=image_loader))
        if name == "GOT10K_train_full":
            if settings.use_lmdb:
                print("Building got10k_train_full from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='train_full', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='train_full', image_loader=image_loader))
        if name == "GOT10K_votval":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='votval', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='votval', image_loader=image_loader))
        if name == "COCO171":
            if settings.use_lmdb:
                print("Building COCO2017 from lmdb")
                datasets.append(MSCOCOSeq_lmdb(settings.env.coco_lmdb_dir, version="2017", image_loader=image_loader))
            else:
                datasets.append(MSCOCOSeq(settings.env.coco_dir, version="2017", image_loader=image_loader))
        if name == "VID":
            if settings.use_lmdb:
                print("Building VID from lmdb")
                datasets.append(ImagenetVID_lmdb(settings.env.imagenet_lmdb_dir, image_loader=image_loader))
            else:
                datasets.append(ImagenetVID(settings.env.imagenet_dir, image_loader=image_loader))
        if name == "TRACKINGNET":
            if settings.use_lmdb:
                print("Building TrackingNet from lmdb")
                datasets.append(TrackingNet_lmdb(settings.env.trackingnet_lmdb_dir, image_loader=image_loader))
            else:
                # raise ValueError("NOW WE CAN ONLY USE TRACKINGNET FROM LMDB")
                datasets.append(TrackingNet(settings.env.trackingnet_dir, image_loader=image_loader))
    return datasets

def run(settings):
    settings.move_data_to_gpu = True
    settings.description = ''
    settings.script_name = 'stark_staf'
    #settings.batch_size = 8
  
    settings.num_workers = 1
    settings.print_interval = 5
    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std = [0.229, 0.224, 0.225]
    settings.search_area_factor = 4
    settings.output_sigma_factor = 1/4
    settings.target_filter_sz = 4
    settings.feature_sz = 20 #18
    settings.use_lmdb = False
    #settings.output_sz = 320 #settings.feature_sz * 16
    #settings.center_jitter_param = {'train_mode': 'uniform', 'train_factor': 3.0, 'train_limit_motion': False,
                                    #'test_mode': 'uniform', 'test_factor': 4, 'test_limit_motion': True}
    #settings.scale_jitter_param = {'train_factor': 0.25, 'test_factor': 0.4}
    settings.hinge_threshold = 0.01
    settings.print_stats = ["Loss/total","Loss/actmap","Loss/state","Loss/box"]
    
    lasot_train = Lasot(settings.env.lasot_dir, split='train')
    got10k_train = Got10k(settings.env.got10k_dir, split='vottrain')
    trackingnet_train = TrackingNet(settings.env.trackingnet_dir, set_ids=[ 0, 1, 2, 3])

    # Validation datasets
    got10k_val = Got10k(settings.env.got10k_dir, split='votval')
    config_module = importlib.import_module("extern.stark.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file('/workspace/ABA/experiments/stark_staf/config.yaml')
    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    #tfm.RandomHorizontalFlip(probability=0.5)
                                    )

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    #tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    
    # update setting
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.search_area_factor = {'template': cfg.DATA.TEMPLATE.FACTOR,
                                   'search': cfg.DATA.SEARCH.FACTOR}
    settings.output_sz = {'template': cfg.DATA.TEMPLATE.SIZE,
                          'search': cfg.DATA.SEARCH.SIZE}
    settings.center_jitter_factor = {'template': cfg.DATA.TEMPLATE.CENTER_JITTER,
                                     'search': cfg.DATA.SEARCH.CENTER_JITTER}
    settings.scale_jitter_factor = {'template': cfg.DATA.TEMPLATE.SCALE_JITTER,
                                    'search': cfg.DATA.SEARCH.SCALE_JITTER}
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM

    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE

    #output_sigma = settings.output_sigma_factor / settings.search_area_factor
    proposal_params = None

    

    data_processing_train = processing.STARKProcessing(search_area_factor=settings.search_area_factor,
                                                       output_sz=settings.output_sz,
                                                       center_jitter_factor=settings.center_jitter_factor,
                                                       scale_jitter_factor=settings.scale_jitter_factor,
                                                       mode='sequence',
                                                       transform=transform_train,
                                                       joint_transform=transform_joint,
                                                       settings=settings)

    data_processing_val = processing.STARKProcessing(search_area_factor=settings.search_area_factor,
                                                     output_sz=settings.output_sz,
                                                     center_jitter_factor=settings.center_jitter_factor,
                                                     scale_jitter_factor=settings.scale_jitter_factor,
                                                     mode='sequence',
                                                     transform=transform_val,
                                                     joint_transform=transform_joint,
                                                     settings=settings)

    

    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 10)
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
    print(cfg)
    dataset_train = sampler.TrackingSampler([got10k_train, trackingnet_train, lasot_train],
                                            [0.3, 0.3, 0.25],
                                            samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                            num_template_frames=settings.num_template, processing=data_processing_train,
                                            frame_sample_mode=sampler_mode, train_cls=train_cls)

    train_sampler = None
    shuffle = True
    
    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=train_sampler)

    # Validation samplers and loaders
    dataset_val = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.VAL.DATASETS_NAME, settings, opencv_loader),
                                          p_datasets=cfg.DATA.VAL.DATASETS_RATIO,
                                          samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
                                          max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                          num_template_frames=settings.num_template, processing=data_processing_val,
                                          frame_sample_mode=sampler_mode, train_cls=train_cls)
    val_sampler = None
    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=cfg.TRAIN.BATCH_SIZE,
                           num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=val_sampler,
                           epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL)

    # load base stark
    
    param_module = importlib.import_module('extern.stark.test.parameter.{}'.format('stark_staf'))
    params = param_module.parameters(None)
    net = build_starkst(params.cfg)
    net.load_state_dict(torch.load(params.checkpoint, map_location='cpu')['net'], strict=False)
    #net = STARK_ST(params)#.cuda().train()
    #for name,parameters in net.named_parameters():
        #print(name,':',parameters.size())

    
    
    
    
    

    # To be safe
    for p in net.backbone.parameters():
        p.requires_grad_(False)
    for p in net.transformer.parameters():
        p.requires_grad_(False)
    for p in net.box_head.parameters():
        p.requires_grad_(False)
    for p in net.query_embed.parameters():
        p.requires_grad_(False)
    for p in net.bottleneck.parameters():
        p.requires_grad_(False)
    for p in net.cls_head.parameters():
        p.requires_grad_(False)

    # train predictor
    for p in net.box_head.predictor.parameters():
        p.requires_grad_(True)
    
    objective =    {'LBHinge':ltr_losses.LBHingev2(threshold=settings.hinge_threshold, return_per_sequence=False),
                    'giou':giou_loss,
                    'l1':   torch.nn.SmoothL1Loss()  #l1_loss
                    } #{'l1':l1_loss}

    loss_weight = {'act_map': 0.005*500,  'state_after_prop': 0.002*500,'iou':2.0 *500}   #

    #dimp_jitter_fn = DiMPScoreJittering(distractor_ratio=0.1, p_distractor=0.3, max_distractor_enhance_factor=1.3,min_distractor_enhance_factor=0.8)
    actor = actors.STARK_STAFActor(net=net, objective=objective, loss_weight=loss_weight,
                            dimp_jitter_fn=None,settings = settings)

    optimizer = optim.Adam([{'params': actor.net.box_head.predictor.parameters(), 'lr': 5e-3}],
                           lr=5e-3)

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.2)

    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

    trainer.train(40, load_latest=True, fail_safe=False)
