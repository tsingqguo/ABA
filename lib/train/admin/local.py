class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/cheng/Stark-main'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/cheng/Stark-main/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/cheng/Stark-main/pretrained_networks'
        self.lasot_dir = '/dataset/lasot'
        self.got10k_dir = '/dataset/got10k'
        self.lasot_lmdb_dir = '/dataset/lasot_lmdb'
        self.got10k_lmdb_dir = '/dataset/got10k_lmdb'
        self.trackingnet_dir = '/dataset/trackingnet'
        self.trackingnet_lmdb_dir = '/dataset/trackingnet_lmdb'
        self.coco_dir = '/dataset/coco'
        self.coco_lmdb_dir = '/dataset/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/dataset/vid'
        self.imagenet_lmdb_dir = '/dataset/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
