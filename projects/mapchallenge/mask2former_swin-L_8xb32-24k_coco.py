_delete_ = True
base_batch_size = 16
auto_scale_lr = dict(base_batch_size=base_batch_size, enable=False)
backbone_embed_multi = dict(decay_mult=0.0, lr_mult=0.1)
backbone_norm_multi = dict(decay_mult=0.0, lr_mult=0.1)
backend_args = None
batch_augments = [
    dict(
        img_pad_value=0,
        mask_pad_value=0,
        pad_mask=True,
        pad_seg=False,
        size=(
            320,
            320,
        ),
        type="BatchFixedSizePad",
    ),
]
interval = 2000
max_iters = 24000
pretrained = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth"
# set all layers in backbone to lr_mult=0.1
# set all norm layers, position_embeding,
# query_embeding, level_embeding to decay_multi=0.0
depths = [2, 2, 18, 2]
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys = {
    "backbone": dict(lr_mult=0.1, decay_mult=1.0),
    "backbone.patch_embed.norm": backbone_norm_multi,
    "backbone.norm": backbone_norm_multi,
    "absolute_pos_embed": backbone_embed_multi,
    "relative_position_bias_table": backbone_embed_multi,
    "query_embed": embed_multi,
    "query_feat": embed_multi,
    "level_embed": embed_multi,
}
custom_keys.update(
    {
        f"backbone.stages.{stage_id}.blocks.{block_id}.norm": backbone_norm_multi
        for stage_id, num_blocks in enumerate(depths)
        for block_id in range(num_blocks)
    }
)
custom_keys.update(
    {
        f"backbone.stages.{stage_id}.downsample.norm": backbone_norm_multi
        for stage_id in range(len(depths) - 1)
    }
)

batch_size = base_batch_size
dataset_type = "SatelliteDataset"
data_root = "/usr/instance_segmentation/"
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=interval,
        max_keep_ckpts=3,
        save_last=True,
        type="CheckpointHook",
        save_best="coco/segm_mAP_50",
        rule="greater",
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
# Visualization settings
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    name="visualizer",
    type="DetLocalVisualizer",
    vis_backends=[dict(type="LocalVisBackend")],
)
mean = [88.03, 104.33, 115.77]
std = [44.37, 43.48, 41.56]
image_size = (
    320,
    320,
)

launcher = "pytorch"
load_from = None
log_level = "INFO"
log_processor = dict(by_epoch=False, type="LogProcessor", window_size=50)

default_scope = "mmdet"
depths = [
    2,
    2,
    18,
    2,
]
dynamic_intervals = [
    (
        # 365001,
        max_iters * 0.96,
        max_iters,
    ),
]
embed_multi = dict(decay_mult=0.0, lr_mult=1.0)
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend="nccl"),
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
)

model = dict(
    backbone=dict(
        attn_drop_rate=0.0,
        convert_weights=True,
        depths=[
            2,
            2,
            18,
            2,
        ],
        drop_path_rate=0.3,
        drop_rate=0.0,
        embed_dims=192,
        frozen_stages=-1,
        init_cfg=dict(
            checkpoint=pretrained,
            type="Pretrained",
        ),
        mlp_ratio=4,
        num_heads=[
            6,
            12,
            24,
            48,
        ],
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        patch_norm=True,
        qk_scale=None,
        qkv_bias=True,
        type="SwinTransformer",
        window_size=12,
        with_cp=False,
    ),
    data_preprocessor=dict(
        batch_augments=[
            dict(
                img_pad_value=0,
                mask_pad_value=0,
                pad_mask=True,
                pad_seg=False,
                size=(
                    320,
                    320,
                ),
                type="BatchFixedSizePad",
            ),
        ],
        bgr_to_rgb=True,
        mask_pad_value=0,
        mean=mean,
        pad_mask=True,
        pad_seg=False,
        pad_size_divisor=32,
        seg_pad_value=255,
        std=std,
        type="DetDataPreprocessor",
    ),
    init_cfg=None,
    panoptic_fusion_head=dict(
        init_cfg=None,
        loss_panoptic=None,
        num_stuff_classes=0,
        num_things_classes=1,
        type="MaskFormerFusionHead",
    ),
    panoptic_head=dict(
        enforce_decoder_input_project=False,
        feat_channels=256,
        in_channels=[192, 384, 768, 1536],  # Removed duplication
        loss_cls=dict(
            class_weight=[1.0, 1.0],
            loss_weight=2.0,
            reduction="mean",
            type="CrossEntropyLoss",
            use_sigmoid=False,
        ),
        loss_dice=dict(
            activate=True,
            eps=1.0,
            loss_weight=5.0,
            naive_dice=True,
            reduction="mean",
            type="DiceLoss",
            use_sigmoid=True,
        ),
        loss_mask=dict(
            loss_weight=5.0, reduction="mean", type="CrossEntropyLoss", use_sigmoid=True
        ),
        num_queries=100,
        num_stuff_classes=0,
        num_things_classes=1,
        num_transformer_feat_level=4,
        out_channels=256,
        pixel_decoder=dict(
            act_cfg=dict(type="ReLU"),
            encoder=dict(
                layer_cfg=dict(
                    ffn_cfg=dict(
                        act_cfg=dict(inplace=True, type="ReLU"),
                        embed_dims=256,
                        feedforward_channels=2048,
                        ffn_drop=0.0,
                        num_fcs=2,
                    ),
                    self_attn_cfg=dict(
                        batch_first=True,
                        dropout=0.0,
                        embed_dims=256,
                        num_heads=8,
                        num_levels=4,  # Ensure this matches the number of feature levels
                        num_points=4,
                    ),
                ),
                num_layers=6,
            ),
            norm_cfg=dict(num_groups=32, type="GN"),
            num_outs=4,  # Ensure this matches the number of output feature levels
            positional_encoding=dict(normalize=True, num_feats=128),
            type="MSDeformAttnPixelDecoder",
        ),
        positional_encoding=dict(normalize=True, num_feats=128),
        strides=[
            4,
            8,
            16,
            32,
        ],
        transformer_decoder=dict(
            init_cfg=None,
            layer_cfg=dict(
                cross_attn_cfg=dict(
                    batch_first=True, dropout=0.0, embed_dims=256, num_heads=8
                ),
                ffn_cfg=dict(
                    act_cfg=dict(inplace=True, type="ReLU"),
                    embed_dims=256,
                    feedforward_channels=2048,
                    ffn_drop=0.0,
                    num_fcs=2,
                ),
                self_attn_cfg=dict(
                    batch_first=True, dropout=0.0, embed_dims=256, num_heads=8
                ),
            ),
            num_layers=9,
            return_intermediate=True,
        ),
        type="Mask2FormerHead",
    ),
    test_cfg=dict(
        filter_low_score=True,
        instance_on=True,
        iou_thr=0.5,
        max_per_image=100,
        panoptic_on=False,
        semantic_on=False,
    ),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type="ClassificationCost", weight=2.0),
                dict(type="CrossEntropyLossCost", use_sigmoid=True, weight=5.0),
                dict(eps=1.0, pred_act=True, type="DiceCost", weight=5.0),
            ],
            type="HungarianAssigner",
        ),
        importance_sample_ratio=0.75,
        num_points=12544,
        oversample_ratio=3.0,
        sampler=dict(type="MaskPseudoSampler"),
    ),
    type="Mask2Former",
)
num_classes = 1
num_stuff_classes = 1
num_things_classes = 1
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.01, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-08,
        lr=0.0001,
        type="AdamW",
        weight_decay=0.05,
    ),
    paramwise_cfg=dict(
        custom_keys=dict(
            {
                "absolute_pos_embed": dict(decay_mult=0.0, lr_mult=0.1),
                "backbone": dict(decay_mult=1.0, lr_mult=0.1),
                "backbone.norm": dict(decay_mult=0.0, lr_mult=0.1),
                "backbone.patch_embed.norm": dict(decay_mult=0.0, lr_mult=0.1),
                "backbone.stages.0.blocks.0.norm": dict(decay_mult=0.0, lr_mult=0.1),
                "backbone.stages.0.blocks.1.norm": dict(decay_mult=0.0, lr_mult=0.1),
                "backbone.stages.0.downsample.norm": dict(decay_mult=0.0, lr_mult=0.1),
                "backbone.stages.1.blocks.0.norm": dict(decay_mult=0.0, lr_mult=0.1),
                "backbone.stages.1.blocks.1.norm": dict(decay_mult=0.0, lr_mult=0.1),
                "backbone.stages.1.downsample.norm": dict(decay_mult=0.0, lr_mult=0.1),
                "backbone.stages.2.blocks.0.norm": dict(decay_mult=0.0, lr_mult=0.1),
                "backbone.stages.2.blocks.1.norm": dict(decay_mult=0.0, lr_mult=0.1),
                "backbone.stages.2.blocks.10.norm": dict(decay_mult=0.0, lr_mult=0.1),
                "backbone.stages.2.blocks.11.norm": dict(decay_mult=0.0, lr_mult=0.1),
                "backbone.stages.2.blocks.12.norm": dict(decay_mult=0.0, lr_mult=0.1),
                "backbone.stages.2.blocks.13.norm": dict(decay_mult=0.0, lr_mult=0.1),
                "backbone.stages.2.blocks.14.norm": dict(decay_mult=0.0, lr_mult=0.1),
                "backbone.stages.2.blocks.15.norm": dict(decay_mult=0.0, lr_mult=0.1),
                "backbone.stages.2.blocks.16.norm": dict(decay_mult=0.0, lr_mult=0.1),
                "backbone.stages.2.blocks.17.norm": dict(decay_mult=0.0, lr_mult=0.1),
                "backbone.stages.2.blocks.2.norm": dict(decay_mult=0.0, lr_mult=0.1),
                "backbone.stages.2.blocks.3.norm": dict(decay_mult=0.0, lr_mult=0.1),
                "backbone.stages.2.blocks.4.norm": dict(decay_mult=0.0, lr_mult=0.1),
                "backbone.stages.2.blocks.5.norm": dict(decay_mult=0.0, lr_mult=0.1),
                "backbone.stages.2.blocks.6.norm": dict(decay_mult=0.0, lr_mult=0.1),
                "backbone.stages.2.blocks.7.norm": dict(decay_mult=0.0, lr_mult=0.1),
                "backbone.stages.2.blocks.8.norm": dict(decay_mult=0.0, lr_mult=0.1),
                "backbone.stages.2.blocks.9.norm": dict(decay_mult=0.0, lr_mult=0.1),
                "backbone.stages.2.downsample.norm": dict(decay_mult=0.0, lr_mult=0.1),
                "backbone.stages.3.blocks.0.norm": dict(decay_mult=0.0, lr_mult=0.1),
                "backbone.stages.3.blocks.1.norm": dict(decay_mult=0.0, lr_mult=0.1),
                "level_embed": dict(decay_mult=0.0, lr_mult=1.0),
                "query_embed": dict(decay_mult=0.0, lr_mult=1.0),
                "query_feat": dict(decay_mult=0.0, lr_mult=1.0),
                "relative_position_bias_table": dict(decay_mult=0.0, lr_mult=0.1),
            }
        ),
        norm_decay_mult=0.0,
    ),
    type="OptimWrapper",
)
param_scheduler = dict(
    begin=0,
    by_epoch=False,
    end=max_iters,
    gamma=0.1,
    milestones=[
        # 327778,
        # 355092,
        max_iters * 0.8,
        max_iters * 0.9,
    ],
    type="MultiStepLR",
)


train_cfg = dict(
    dynamic_intervals=[
        (
            max_iters * 0.96,
            max_iters,
        ),
    ],
    max_iters=max_iters,
    type="IterBasedTrainLoop",
    val_interval=interval,
)


work_dir = "./work_dirs/mask2former_swin-L_8xb32-25k_satellite"


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
    sampler=dict(type="InfiniteSampler", shuffle=True),
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
        ann_file="merged_val_split/annotations/annotation_new.json",
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
        ann_file="merged_test_split/annotations/annotation_new.json",
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
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
work_dir = "./work_dirs/mask2former_swin-L_8xb32-24k_coco"
