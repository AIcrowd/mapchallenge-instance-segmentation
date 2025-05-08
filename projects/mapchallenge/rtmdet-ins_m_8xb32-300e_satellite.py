auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
base_lr = 0.004
batch_size = 32  # Added batch_size variable
dataset_type = "SatelliteDataset"
data_root = "/usr/instance_segmentation/"
default_scope = "mmdet"
max_epochs = 300

# Added TTA model configuration
tta_model = dict(
    type="DetTTAModel",
    tta_cfg=dict(nms=dict(type="nms", iou_threshold=0.5), max_per_img=100),
)

# Added TTA pipeline
tta_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(
        type="TestTimeAug",
        transforms=[
            [
                dict(type="Resize", scale=(320, 320), keep_ratio=True),
            ],
            [
                dict(type="RandomFlip", prob=1.0),
                dict(type="RandomFlip", prob=0.0),
            ],
            [
                dict(type="Pad", size=(320, 320), pad_val=dict(img=(114, 114, 114))),
            ],
            [
                dict(
                    type="PackDetInputs",
                    meta_keys=(
                        "img_id",
                        "img_path",
                        "ori_shape",
                        "img_shape",
                        "scale_factor",
                        "flip",
                        "flip_direction",
                    ),
                )
            ],
        ],
    ),
]

custom_hooks = [
    dict(
        ema_type="ExpMomentumEMA",
        momentum=0.0002,
        priority=49,
        type="EMAHook",
        update_buffers=True,
    ),
    dict(
        switch_epoch=280,
        switch_pipeline=[
            dict(backend_args=None, type="LoadImageFromFile"),
            dict(
                poly2mask=False, type="LoadAnnotations", with_bbox=True, with_mask=True
            ),
            dict(
                keep_ratio=True,
                ratio_range=(0.1, 2.0),
                scale=(320, 320),
                type="RandomResize",
            ),
            dict(
                allow_negative_crop=True,
                crop_size=(320, 320),
                recompute_bbox=True,
                type="RandomCrop",
            ),
            dict(min_gt_bbox_wh=(1, 1), type="FilterAnnotations"),
            dict(type="YOLOXHSVRandomAug"),
            dict(prob=0.5, type="RandomFlip"),
            dict(pad_val=dict(img=(114, 114, 114)), size=(320, 320), type="Pad"),
            dict(type="PackDetInputs"),
        ],
        type="PipelineSwitchHook",
    ),
]

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        save_best="coco/segm_mAP_50",
        rule="greater",
        max_keep_ckpts=3,
    ),
    early_stopping=dict(
        type="EarlyStoppingHook",
        monitor="coco/segm_mAP_50",
        patience=6,
        min_delta=0.005,
    ),
    logger=dict(interval=50, type="LoggerHook"),
    param_scheduler=dict(type="ParamSchedulerHook"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    timer=dict(type="IterTimerHook"),
    visualization=dict(type="DetVisualizationHook"),
)

mean = [88.03, 104.33, 115.77]
std = [44.37, 43.48, 41.56]
# Model configuration
model = dict(
    backbone=dict(
        act_cfg=dict(inplace=True, type="SiLU"),
        arch="P5",
        channel_attention=True,
        deepen_factor=0.67,
        expand_ratio=0.5,
        norm_cfg=dict(type="SyncBN"),
        type="CSPNeXt",
        widen_factor=0.75,
    ),
    bbox_head=dict(
        act_cfg=dict(inplace=True, type="SiLU"),
        anchor_generator=dict(offset=0, strides=[8, 16, 32], type="MlvlPointGenerator"),
        bbox_coder=dict(type="DistancePointBBoxCoder"),
        feat_channels=192,
        in_channels=192,
        loss_bbox=dict(loss_weight=2.0, type="GIoULoss"),
        loss_cls=dict(
            beta=2.0, loss_weight=1.0, type="QualityFocalLoss", use_sigmoid=True
        ),
        loss_mask=dict(eps=5e-06, loss_weight=2.0, reduction="mean", type="DiceLoss"),
        norm_cfg=dict(requires_grad=True, type="SyncBN"),
        num_classes=1,
        pred_kernel_size=1,
        share_conv=True,
        stacked_convs=2,
        type="RTMDetInsSepBNHead",
    ),
    data_preprocessor=dict(
        batch_augments=None,
        bgr_to_rgb=False,
        mean=mean,
        std=std,
        type="DetDataPreprocessor",
    ),
    neck=dict(
        act_cfg=dict(inplace=True, type="SiLU"),
        expand_ratio=0.5,
        in_channels=[192, 384, 768],
        norm_cfg=dict(type="SyncBN"),
        num_csp_blocks=2,
        out_channels=192,
        type="CSPNeXtPAFPN",
    ),
    test_cfg=dict(
        mask_thr_binary=0.5,
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.5, type="nms"),
        nms_pre=1000,
        score_thr=0.05,
    ),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(topk=13, type="DynamicSoftLabelAssigner"),
        debug=False,
        pos_weight=-1,
    ),
    type="RTMDet",
)

# Training configuration
train_pipeline = [
    dict(backend_args=None, type="LoadImageFromFile"),
    dict(poly2mask=False, type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(img_scale=(320, 320), pad_val=114.0, type="CachedMosaic"),
    dict(
        keep_ratio=True, ratio_range=(0.1, 2.0), scale=(320, 320), type="RandomResize"
    ),
    dict(
        allow_negative_crop=True,
        crop_size=(320, 320),
        recompute_bbox=True,
        type="RandomCrop",
    ),
    dict(type="YOLOXHSVRandomAug"),
    dict(prob=0.5, type="RandomFlip"),
    dict(pad_val=dict(img=(114, 114, 114)), size=(320, 320), type="Pad"),
    dict(
        img_scale=(320, 320),
        max_cached_images=20,
        pad_val=(114, 114, 114),
        ratio_range=(1.0, 1.0),
        type="CachedMixUp",
    ),
    dict(min_gt_bbox_wh=(1, 1), type="FilterAnnotations"),
    dict(type="PackDetInputs"),
]

test_pipeline = [
    dict(backend_args=None, type="LoadImageFromFile"),
    dict(keep_ratio=True, scale=(320, 320), type="Resize"),
    dict(pad_val=dict(img=(114, 114, 114)), size=(320, 320), type="Pad"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
        type="PackDetInputs",
    ),
]

# Dataloader configurations
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="merged_train_split/annotations/annotation_new.json",
        data_prefix=dict(img="merged_train_split/images/"),
        filter_cfg=dict(filter_empty_gt=True, min_size=15),
        pipeline=train_pipeline,
        backend_args=None,
    ),
)

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="merged_val_split/annotations/annotations.json",
        data_prefix=dict(img="merged_val_split/images"),
        filter_cfg=dict(filter_empty_gt=True, min_size=15),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=None,
    ),
)

test_dataloader = dict(
    batch_size=batch_size,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="merged_test_split/annotations/annotations.json",
        data_prefix=dict(img="merged_test_split/images/"),
        filter_cfg=dict(filter_empty_gt=True, min_size=15),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=None,
    ),
)

# Evaluators
val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "merged_val_split/annotations/annotations.json",
    metric=["segm"],
    format_only=False,
    backend_args=None,
)

test_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "merged_test_split/annotations/annotations.json",
    metric=["segm"],
    format_only=False,
    backend_args=None,
)

# Other configurations
default_scope = "mmdet"
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend="nccl"),
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
)

train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=300,
    val_interval=10,
    dynamic_intervals=[(280, 1)],
)

optim_wrapper = dict(
    optimizer=dict(lr=0.004, type="AdamW", weight_decay=0.05),
    paramwise_cfg=dict(bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type="OptimWrapper",
)

param_scheduler = [
    dict(begin=0, by_epoch=False, end=1000, start_factor=1e-05, type="LinearLR"),
    dict(
        T_max=150,
        begin=150,
        by_epoch=True,
        convert_to_iter_based=True,
        end=300,
        eta_min=0.0002,
        type="CosineAnnealingLR",
    ),
]

# Visualization settings
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    name="visualizer",
    type="DetLocalVisualizer",
    vis_backends=[dict(type="LocalVisBackend")],
)

# Runtime settings
log_level = "INFO"
load_from = "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_m_8xb32-300e_coco/rtmdet-ins_m_8xb32-300e_coco_20221123_001039-6eba602e.pth"
resume = False
work_dir = "./work_dirs/rtmdet-ins_m_8xb32-300e_satellite"
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
