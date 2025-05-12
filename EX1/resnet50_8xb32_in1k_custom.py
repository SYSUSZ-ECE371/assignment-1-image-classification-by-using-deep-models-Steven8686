_base_ = [
    '../_base_/models/resnet50.py', '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]
model = dict(
    backbone=dict(
        frozen_stages=2,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
            prefix='backbone',
        )),
    head=dict(num_classes=5),
)
data_root = '/home/caizhenyu/ruimao_classes/assignment1/data'
train_dataloader = dict(
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        ann_file='',       # 我们假定使用子文件夹格式，因此需要将标注文件置空
        data_prefix='train',
    ))
val_dataloader = dict(
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        ann_file='',       # 我们假定使用子文件夹格式，因此需要将标注文件置空
        data_prefix='test',
    ))
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
# 学习率策略
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[15], gamma=0.1)