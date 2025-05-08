_base_ = "./rtmdet-ins_m_8xb32-300e_satellite.py"

model = dict(
    backbone=dict(deepen_factor=1.33, widen_factor=1.25),
    neck=dict(in_channels=[320, 640, 1280], out_channels=320, num_csp_blocks=4),
    bbox_head=dict(in_channels=320, feat_channels=320),
)

base_lr = 0.002

# optimizer
optim_wrapper = dict(optimizer=dict(lr=base_lr))


# learning rate
param_scheduler = [
    dict(type="LinearLR", start_factor=1.0e-5, by_epoch=False, begin=0, end=1000),
    dict(
        # use cosine lr from 150 to 300 epoch
        type="CosineAnnealingLR",
        eta_min=base_lr * 0.05,
        begin=_base_.max_epochs // 2,
        end=_base_.max_epochs,
        T_max=_base_.max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_x_8xb16-300e_coco/rtmdet-ins_x_8xb16-300e_coco_20221124_111313-33d4595b.pth'
work_dir = "./work_dirs/rtmdet-ins_X_8xb16-300e_satellite"
