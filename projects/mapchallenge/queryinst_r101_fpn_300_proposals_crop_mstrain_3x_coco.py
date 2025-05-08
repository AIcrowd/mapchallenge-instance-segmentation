_base_ = "./queryinst_r50_fpn_ms-3x_coco.py"

num_proposals = 300


model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet101"),
    ),
    rpn_head=dict(num_proposals=num_proposals),
    test_cfg=dict(
        _delete_=True,
        rpn=None,
        rcnn=dict(max_per_img=num_proposals, mask_thr_binary=0.5),
    ),
)
work_dir = "./work_dirs/queryinst_r101_fpn_300_proposals_crop_mstrain_2x_coco"
load_from = "https://download.openmmlab.com/mmdetection/v2.0/queryinst/queryinst_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco/queryinst_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20210904_153621-76cce59f.pth"
