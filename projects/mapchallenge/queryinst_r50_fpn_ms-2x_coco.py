batch_size = 16
auto_scale_lr = dict(base_batch_size=batch_size, enable=False)
backend_args = None
resume = False
max_epochs = 24
data_root = "/usr/instance_segmentation/"
default_scope = "mmdet"
dataset_type = "SatelliteDataset"
launcher = "pytorch"
load_from = "https://download.openmmlab.com/mmdetection/v2.0/queryinst/queryinst_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco/queryinst_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20210904_153621-76cce59f.pth"
log_level = "INFO"

mean = [88.03, 104.33, 115.77]
std = [44.37, 43.48, 41.56]
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        type="CheckpointHook",
        save_best="coco/segm_mAP_50",
        rule="greater",
        max_keep_ckpts=3,
    ),
    logger=dict(interval=50, type="LoggerHook"),
    param_scheduler=dict(type="ParamSchedulerHook"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    timer=dict(type="IterTimerHook"),
    visualization=dict(type="DetVisualizationHook"),
)

env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend="nccl"),
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
)

log_processor = dict(by_epoch=True, type="LogProcessor", window_size=50)
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint="torchvision://resnet50", type="Pretrained"),
        norm_cfg=dict(requires_grad=True, type="BN"),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style="pytorch",
        type="ResNet",
    ),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=mean,
        pad_mask=True,
        pad_size_divisor=32,
        std=std,
        type="DetDataPreprocessor",
    ),
    neck=dict(
        add_extra_convs="on_input",
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=4,
        out_channels=256,
        start_level=0,
        type="FPN",
    ),
    roi_head=dict(
        bbox_head=[
            dict(
                bbox_coder=dict(
                    clip_border=False,
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.5,
                        0.5,
                        1.0,
                        1.0,
                    ],
                    type="DeltaXYWHBBoxCoder",
                ),
                dropout=0.0,
                dynamic_conv_cfg=dict(
                    act_cfg=dict(inplace=True, type="ReLU"),
                    feat_channels=64,
                    in_channels=256,
                    input_feat_shape=7,
                    norm_cfg=dict(type="LN"),
                    out_channels=256,
                    type="DynamicConv",
                ),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type="ReLU"),
                in_channels=256,
                loss_bbox=dict(loss_weight=5.0, type="L1Loss"),
                loss_cls=dict(
                    alpha=0.25,
                    gamma=2.0,
                    loss_weight=2.0,
                    type="FocalLoss",
                    use_sigmoid=True,
                ),
                loss_iou=dict(loss_weight=2.0, type="GIoULoss"),
                num_classes=1,
                num_cls_fcs=1,
                num_ffn_fcs=2,
                num_heads=8,
                num_reg_fcs=3,
                type="DIIHead",
            ),
            dict(
                bbox_coder=dict(
                    clip_border=False,
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.5,
                        0.5,
                        1.0,
                        1.0,
                    ],
                    type="DeltaXYWHBBoxCoder",
                ),
                dropout=0.0,
                dynamic_conv_cfg=dict(
                    act_cfg=dict(inplace=True, type="ReLU"),
                    feat_channels=64,
                    in_channels=256,
                    input_feat_shape=7,
                    norm_cfg=dict(type="LN"),
                    out_channels=256,
                    type="DynamicConv",
                ),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type="ReLU"),
                in_channels=256,
                loss_bbox=dict(loss_weight=5.0, type="L1Loss"),
                loss_cls=dict(
                    alpha=0.25,
                    gamma=2.0,
                    loss_weight=2.0,
                    type="FocalLoss",
                    use_sigmoid=True,
                ),
                loss_iou=dict(loss_weight=2.0, type="GIoULoss"),
                num_classes=1,
                num_cls_fcs=1,
                num_ffn_fcs=2,
                num_heads=8,
                num_reg_fcs=3,
                type="DIIHead",
            ),
            dict(
                bbox_coder=dict(
                    clip_border=False,
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.5,
                        0.5,
                        1.0,
                        1.0,
                    ],
                    type="DeltaXYWHBBoxCoder",
                ),
                dropout=0.0,
                dynamic_conv_cfg=dict(
                    act_cfg=dict(inplace=True, type="ReLU"),
                    feat_channels=64,
                    in_channels=256,
                    input_feat_shape=7,
                    norm_cfg=dict(type="LN"),
                    out_channels=256,
                    type="DynamicConv",
                ),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type="ReLU"),
                in_channels=256,
                loss_bbox=dict(loss_weight=5.0, type="L1Loss"),
                loss_cls=dict(
                    alpha=0.25,
                    gamma=2.0,
                    loss_weight=2.0,
                    type="FocalLoss",
                    use_sigmoid=True,
                ),
                loss_iou=dict(loss_weight=2.0, type="GIoULoss"),
                num_classes=1,
                num_cls_fcs=1,
                num_ffn_fcs=2,
                num_heads=8,
                num_reg_fcs=3,
                type="DIIHead",
            ),
            dict(
                bbox_coder=dict(
                    clip_border=False,
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.5,
                        0.5,
                        1.0,
                        1.0,
                    ],
                    type="DeltaXYWHBBoxCoder",
                ),
                dropout=0.0,
                dynamic_conv_cfg=dict(
                    act_cfg=dict(inplace=True, type="ReLU"),
                    feat_channels=64,
                    in_channels=256,
                    input_feat_shape=7,
                    norm_cfg=dict(type="LN"),
                    out_channels=256,
                    type="DynamicConv",
                ),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type="ReLU"),
                in_channels=256,
                loss_bbox=dict(loss_weight=5.0, type="L1Loss"),
                loss_cls=dict(
                    alpha=0.25,
                    gamma=2.0,
                    loss_weight=2.0,
                    type="FocalLoss",
                    use_sigmoid=True,
                ),
                loss_iou=dict(loss_weight=2.0, type="GIoULoss"),
                num_classes=1,
                num_cls_fcs=1,
                num_ffn_fcs=2,
                num_heads=8,
                num_reg_fcs=3,
                type="DIIHead",
            ),
            dict(
                bbox_coder=dict(
                    clip_border=False,
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.5,
                        0.5,
                        1.0,
                        1.0,
                    ],
                    type="DeltaXYWHBBoxCoder",
                ),
                dropout=0.0,
                dynamic_conv_cfg=dict(
                    act_cfg=dict(inplace=True, type="ReLU"),
                    feat_channels=64,
                    in_channels=256,
                    input_feat_shape=7,
                    norm_cfg=dict(type="LN"),
                    out_channels=256,
                    type="DynamicConv",
                ),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type="ReLU"),
                in_channels=256,
                loss_bbox=dict(loss_weight=5.0, type="L1Loss"),
                loss_cls=dict(
                    alpha=0.25,
                    gamma=2.0,
                    loss_weight=2.0,
                    type="FocalLoss",
                    use_sigmoid=True,
                ),
                loss_iou=dict(loss_weight=2.0, type="GIoULoss"),
                num_classes=1,
                num_cls_fcs=1,
                num_ffn_fcs=2,
                num_heads=8,
                num_reg_fcs=3,
                type="DIIHead",
            ),
            dict(
                bbox_coder=dict(
                    clip_border=False,
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.5,
                        0.5,
                        1.0,
                        1.0,
                    ],
                    type="DeltaXYWHBBoxCoder",
                ),
                dropout=0.0,
                dynamic_conv_cfg=dict(
                    act_cfg=dict(inplace=True, type="ReLU"),
                    feat_channels=64,
                    in_channels=256,
                    input_feat_shape=7,
                    norm_cfg=dict(type="LN"),
                    out_channels=256,
                    type="DynamicConv",
                ),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type="ReLU"),
                in_channels=256,
                loss_bbox=dict(loss_weight=5.0, type="L1Loss"),
                loss_cls=dict(
                    alpha=0.25,
                    gamma=2.0,
                    loss_weight=2.0,
                    type="FocalLoss",
                    use_sigmoid=True,
                ),
                loss_iou=dict(loss_weight=2.0, type="GIoULoss"),
                num_classes=1,
                num_cls_fcs=1,
                num_ffn_fcs=2,
                num_heads=8,
                num_reg_fcs=3,
                type="DIIHead",
            ),
        ],
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=2, type="RoIAlign"),
            type="SingleRoIExtractor",
        ),
        mask_head=[
            dict(
                class_agnostic=False,
                conv_kernel_size=3,
                conv_out_channels=256,
                dynamic_conv_cfg=dict(
                    act_cfg=dict(inplace=True, type="ReLU"),
                    feat_channels=64,
                    in_channels=256,
                    input_feat_shape=14,
                    norm_cfg=dict(type="LN"),
                    out_channels=256,
                    type="DynamicConv",
                    with_proj=False,
                ),
                in_channels=256,
                loss_mask=dict(
                    activate=False,
                    eps=1e-05,
                    loss_weight=8.0,
                    type="DiceLoss",
                    use_sigmoid=True,
                ),
                norm_cfg=dict(type="BN"),
                num_classes=1,
                num_convs=4,
                roi_feat_size=14,
                type="DynamicMaskHead",
                upsample_cfg=dict(scale_factor=2, type="deconv"),
            ),
            dict(
                class_agnostic=False,
                conv_kernel_size=3,
                conv_out_channels=256,
                dynamic_conv_cfg=dict(
                    act_cfg=dict(inplace=True, type="ReLU"),
                    feat_channels=64,
                    in_channels=256,
                    input_feat_shape=14,
                    norm_cfg=dict(type="LN"),
                    out_channels=256,
                    type="DynamicConv",
                    with_proj=False,
                ),
                in_channels=256,
                loss_mask=dict(
                    activate=False,
                    eps=1e-05,
                    loss_weight=8.0,
                    type="DiceLoss",
                    use_sigmoid=True,
                ),
                norm_cfg=dict(type="BN"),
                num_classes=1,
                num_convs=4,
                roi_feat_size=14,
                type="DynamicMaskHead",
                upsample_cfg=dict(scale_factor=2, type="deconv"),
            ),
            dict(
                class_agnostic=False,
                conv_kernel_size=3,
                conv_out_channels=256,
                dynamic_conv_cfg=dict(
                    act_cfg=dict(inplace=True, type="ReLU"),
                    feat_channels=64,
                    in_channels=256,
                    input_feat_shape=14,
                    norm_cfg=dict(type="LN"),
                    out_channels=256,
                    type="DynamicConv",
                    with_proj=False,
                ),
                in_channels=256,
                loss_mask=dict(
                    activate=False,
                    eps=1e-05,
                    loss_weight=8.0,
                    type="DiceLoss",
                    use_sigmoid=True,
                ),
                norm_cfg=dict(type="BN"),
                num_classes=1,
                num_convs=4,
                roi_feat_size=14,
                type="DynamicMaskHead",
                upsample_cfg=dict(scale_factor=2, type="deconv"),
            ),
            dict(
                class_agnostic=False,
                conv_kernel_size=3,
                conv_out_channels=256,
                dynamic_conv_cfg=dict(
                    act_cfg=dict(inplace=True, type="ReLU"),
                    feat_channels=64,
                    in_channels=256,
                    input_feat_shape=14,
                    norm_cfg=dict(type="LN"),
                    out_channels=256,
                    type="DynamicConv",
                    with_proj=False,
                ),
                in_channels=256,
                loss_mask=dict(
                    activate=False,
                    eps=1e-05,
                    loss_weight=8.0,
                    type="DiceLoss",
                    use_sigmoid=True,
                ),
                norm_cfg=dict(type="BN"),
                num_classes=1,
                num_convs=4,
                roi_feat_size=14,
                type="DynamicMaskHead",
                upsample_cfg=dict(scale_factor=2, type="deconv"),
            ),
            dict(
                class_agnostic=False,
                conv_kernel_size=3,
                conv_out_channels=256,
                dynamic_conv_cfg=dict(
                    act_cfg=dict(inplace=True, type="ReLU"),
                    feat_channels=64,
                    in_channels=256,
                    input_feat_shape=14,
                    norm_cfg=dict(type="LN"),
                    out_channels=256,
                    type="DynamicConv",
                    with_proj=False,
                ),
                in_channels=256,
                loss_mask=dict(
                    activate=False,
                    eps=1e-05,
                    loss_weight=8.0,
                    type="DiceLoss",
                    use_sigmoid=True,
                ),
                norm_cfg=dict(type="BN"),
                num_classes=1,
                num_convs=4,
                roi_feat_size=14,
                type="DynamicMaskHead",
                upsample_cfg=dict(scale_factor=2, type="deconv"),
            ),
            dict(
                class_agnostic=False,
                conv_kernel_size=3,
                conv_out_channels=256,
                dynamic_conv_cfg=dict(
                    act_cfg=dict(inplace=True, type="ReLU"),
                    feat_channels=64,
                    in_channels=256,
                    input_feat_shape=14,
                    norm_cfg=dict(type="LN"),
                    out_channels=256,
                    type="DynamicConv",
                    with_proj=False,
                ),
                in_channels=256,
                loss_mask=dict(
                    activate=False,
                    eps=1e-05,
                    loss_weight=8.0,
                    type="DiceLoss",
                    use_sigmoid=True,
                ),
                norm_cfg=dict(type="BN"),
                num_classes=1,
                num_convs=4,
                roi_feat_size=14,
                type="DynamicMaskHead",
                upsample_cfg=dict(scale_factor=2, type="deconv"),
            ),
        ],
        mask_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=14, sampling_ratio=2, type="RoIAlign"),
            type="SingleRoIExtractor",
        ),
        num_stages=6,
        proposal_feature_channel=256,
        stage_loss_weights=[
            1,
            1,
            1,
            1,
            1,
            1,
        ],
        type="SparseRoIHead",
    ),
    rpn_head=dict(
        num_proposals=100, proposal_feature_channel=256, type="EmbeddingRPNHead"
    ),
    test_cfg=dict(rcnn=dict(mask_thr_binary=0.5, max_per_img=100), rpn=None),
    train_cfg=dict(
        rcnn=[
            dict(
                assigner=dict(
                    match_costs=[
                        dict(type="FocalLossCost", weight=2.0),
                        dict(box_format="xyxy", type="BBoxL1Cost", weight=5.0),
                        dict(iou_mode="giou", type="IoUCost", weight=2.0),
                    ],
                    type="HungarianAssigner",
                ),
                mask_size=28,
                pos_weight=1,
                sampler=dict(type="PseudoSampler"),
            ),
            dict(
                assigner=dict(
                    match_costs=[
                        dict(type="FocalLossCost", weight=2.0),
                        dict(box_format="xyxy", type="BBoxL1Cost", weight=5.0),
                        dict(iou_mode="giou", type="IoUCost", weight=2.0),
                    ],
                    type="HungarianAssigner",
                ),
                mask_size=28,
                pos_weight=1,
                sampler=dict(type="PseudoSampler"),
            ),
            dict(
                assigner=dict(
                    match_costs=[
                        dict(type="FocalLossCost", weight=2.0),
                        dict(box_format="xyxy", type="BBoxL1Cost", weight=5.0),
                        dict(iou_mode="giou", type="IoUCost", weight=2.0),
                    ],
                    type="HungarianAssigner",
                ),
                mask_size=28,
                pos_weight=1,
                sampler=dict(type="PseudoSampler"),
            ),
            dict(
                assigner=dict(
                    match_costs=[
                        dict(type="FocalLossCost", weight=2.0),
                        dict(box_format="xyxy", type="BBoxL1Cost", weight=5.0),
                        dict(iou_mode="giou", type="IoUCost", weight=2.0),
                    ],
                    type="HungarianAssigner",
                ),
                mask_size=28,
                pos_weight=1,
                sampler=dict(type="PseudoSampler"),
            ),
            dict(
                assigner=dict(
                    match_costs=[
                        dict(type="FocalLossCost", weight=2.0),
                        dict(box_format="xyxy", type="BBoxL1Cost", weight=5.0),
                        dict(iou_mode="giou", type="IoUCost", weight=2.0),
                    ],
                    type="HungarianAssigner",
                ),
                mask_size=28,
                pos_weight=1,
                sampler=dict(type="PseudoSampler"),
            ),
            dict(
                assigner=dict(
                    match_costs=[
                        dict(type="FocalLossCost", weight=2.0),
                        dict(box_format="xyxy", type="BBoxL1Cost", weight=5.0),
                        dict(iou_mode="giou", type="IoUCost", weight=2.0),
                    ],
                    type="HungarianAssigner",
                ),
                mask_size=28,
                pos_weight=1,
                sampler=dict(type="PseudoSampler"),
            ),
        ],
        rpn=None,
    ),
    type="QueryInst",
)
num_proposals = 100
num_stages = 6

optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(lr=0.0001, type="AdamW", weight_decay=0.0001),
    paramwise_cfg=dict(custom_keys=dict(backbone=dict(decay_mult=1.0, lr_mult=0.1))),
    type="OptimWrapper",
)
param_scheduler = [
    dict(begin=0, by_epoch=False, end=500, start_factor=0.001, type="LinearLR"),
    dict(
        begin=0,
        by_epoch=True,
        end=max_epochs,
        gamma=0.1,
        milestones=[
            int(max_epochs * 0.85),
            int(max_epochs * 0.95),
        ],
        type="MultiStepLR",
    ),
]


test_cfg = dict(type="TestLoop")
val_cfg = dict(type="ValLoop")
train_cfg = dict(max_epochs=max_epochs, type="EpochBasedTrainLoop", val_interval=1)

# Training configuration
train_pipeline = [
    dict(backend_args=None, type="LoadImageFromFile"),
    dict(poly2mask=False, type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        keep_ratio=True,
        scales=[
            (
                320,
                320,
            ),
            (
                256,
                256,
            ),
        ],
        type="RandomChoiceResize",
    ),
    dict(prob=0.5, type="RandomFlip"),
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
        ann_file="merged_train_split/annotations/annotations.json",
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
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    name="visualizer",
    type="DetLocalVisualizer",
    vis_backends=[dict(type="LocalVisBackend")],
)
work_dir = "./work_dirs/queryinst_r50_fpn_ms-2x_coco"
