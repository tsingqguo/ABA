class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = ''    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.lasot_dir = '/dataset/lasot/LaSOTBenchmark'
        self.got10k_dir = '/dataset/training_dataset/got10/train'
        self.trackingnet_dir = '/dataset/training_dataset/trackingnet/TrackingNet'
        self.coco_dir = ''
        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
