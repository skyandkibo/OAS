optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[24, 33])
total_epochs = 36
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
model = dict(
    type='OrientedRCNN',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='SRPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='S2MidpointOffsetCoder',
            target_means=[0.0, 0.0, 0.0, 0.0, 0.0],
            target_stds=[0.75, 0.75, 0.75, 0.75, 0.75]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='OBBStandardRoIHead',
        bbox_roi_extractor=dict(
            type='OBBSingleRoIExtractor',
            roi_layer=dict(type='RoIAlignRotated', out_size=7, sample_num=2),
            out_channels=256,
            extend_factor=(1.4, 1.2),
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='OBBShared2FCBBoxHead',
            start_bbox_type='obb',
            end_bbox_type='obb',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=15,
            bbox_coder=dict(
                type='S2OBB2OBBDeltaXYWHTCoder',
                target_means=[0.0, 0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.5),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
            loss_angle_flag=True,
            loss_angle=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))))
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            match_low_quality=True,
            gpu_assign_thr=200,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.8,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='AngleAssignerOBB',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            match_low_quality=False,
            u1=0.05,
            debug=False,
            angle_type='angle_matrix_func_gt_max',
            iou_calculator=dict(type='OBBOverlaps'),
            ignore_iof_thr=-1),
        sampler=dict(
            type='OBBRandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.8,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='obb_nms', iou_thr=0.1),
        max_per_img=2000))
dataset_type = 'DOTADataset'
data_root = '../DOTAData/ms/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadOBBAnnotations',
        with_bbox=True,
        with_label=True,
        with_poly_as_mask=True),
    dict(type='LoadDOTASpecialInfo'),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='OBBRandomFlip', h_flip_ratio=0.5, v_flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(
        type='RandomOBBRotate',
        rotate_after_flip=True,
        angles=(0, 90),
        vert_rate=0.5,
        vert_cls=['roundabout', 'storage-tank']),
    dict(type='Pad', size_divisor=32),
    dict(type='DOTASpecialIgnore', ignore_size=2),
    dict(type='FliterEmpty'),
    dict(type='Mask2OBB', obb_type='obb'),
    dict(type='OBBDefaultFormatBundle'),
    dict(
        type='OBBCollect',
        keys=['img', 'gt_bboxes', 'gt_obboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipRotateAug',
        img_scale=[(1024, 1024)],
        h_flip=False,
        v_flip=False,
        rotate=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='OBBRandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='RandomOBBRotate', rotate_after_flip=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='OBBCollect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='DOTADataset',
        task='Task1',
        ann_file=data_root+'trainval/annfiles/',
        img_prefix=data_root+'trainval/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='LoadOBBAnnotations',
                with_bbox=True,
                with_label=True,
                with_poly_as_mask=True),
            dict(type='LoadDOTASpecialInfo'),
            dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
            dict(type='OBBRandomFlip', h_flip_ratio=0.5, v_flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='RandomOBBRotate',
                rotate_after_flip=True,
                angles=(0, 90),
                vert_rate=0.5,
                vert_cls=['roundabout', 'storage-tank']),
            dict(type='Pad', size_divisor=32),
            dict(type='DOTASpecialIgnore', ignore_size=2),
            dict(type='FliterEmpty'),
            dict(type='Mask2OBB', obb_type='obb'),
            dict(type='OBBDefaultFormatBundle'),
            dict(
                type='OBBCollect',
                keys=['img', 'gt_bboxes', 'gt_obboxes', 'gt_labels'])
        ]),
    test=dict(
        type='DOTADataset',
        task='Task1',
        ann_file=data_root+'test/annfiles/',
        img_prefix=data_root+'test/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipRotateAug',
                img_scale=[(1024, 1024)],
                h_flip=False,
                v_flip=False,
                rotate=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='OBBRandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='RandomOBBRotate', rotate_after_flip=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='OBBCollect', keys=['img'])
                ])
        ]))
evaluation = None
work_dir = './work_dirs/SR2_ORCNN_r50_dota10_ms_oas'
gpu_ids = range(0, 1)
