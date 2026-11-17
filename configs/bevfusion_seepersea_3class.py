"""Overlay for the native three-class SeaPersea BEVFusion baseline.

This file has no cross-directory ``_base_``. The run tool loads the upstream
fusion config first and merges this file with MMEngine's config merge;
``_delete_=True`` clears inherited dicts that must be replaced as a whole.
"""
point_cloud_range = [-108.0, -108.0, -20.0, 108.0, 108.0, 20.0]
voxel_size = [0.15, 0.15, 1.0]
classes = ('ship', 'buoy', 'other')
data_prefix = dict(pts='samples/LIDAR_TOP', CAM_FRONT='samples/CAM_FRONT')
input_modality = dict(use_lidar=True, use_camera=True)
custom_imports = dict(
    imports=['projects.BEVFusion.bevfusion', 'usv_fusion_ext'],
    allow_failed_imports=False)

model = dict(
    data_preprocessor=dict(
        voxelize_cfg=dict(
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size)),
    img_backbone=dict(init_cfg=None),
    pts_middle_encoder=dict(sparse_shape=[1440, 1440, 41]),
    view_transform=dict(
        image_size=[256, 352],
        feature_size=[32, 44],
        xbound=[-108.0, 108.0, 0.6],
        ybound=[-108.0, 108.0, 0.6],
        zbound=[-20.0, 20.0, 40.0],
        dbound=[1.0, 60.0, 0.5],
        downsample=2),
    bbox_head=dict(
        num_classes=3,
        num_proposals=200,
        nms_kernel_size=3,
        common_heads=dict(
            _delete_=True,
            center=[2, 2],
            height=[1, 2],
            dim=[3, 2],
            rot=[2, 2]),
        train_cfg=dict(
            _delete_=True,
            dataset='SeaPersea',
            point_cloud_range=point_cloud_range,
            grid_size=[1440, 1440, 40],
            voxel_size=voxel_size,
            out_size_factor=8,
            gaussian_overlap=0.1,
            min_radius=2,
            pos_weight=-1,
            code_weights=[1.0] * 8,
            assigner=dict(
                type='HungarianAssigner3D',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(
                    type='mmdet.FocalLossCost',
                    gamma=2.0,
                    alpha=0.25,
                    weight=0.15),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                iou_cost=dict(type='IoU3DCost', weight=0.25))),
        test_cfg=dict(
            _delete_=True,
            dataset='SeaPersea',
            grid_size=[1440, 1440, 40],
            out_size_factor=8,
            voxel_size=[0.15, 0.15],
            pc_range=[-108.0, -108.0],
            nms_type=None),
        bbox_coder=dict(
            _delete_=True,
            type='TransFusionBBoxCoder',
            pc_range=[-108.0, -108.0],
            post_center_range=[-108.0, -108.0, -40.0, 108.0, 108.0, 40.0],
            score_threshold=0.0,
            out_size_factor=8,
            voxel_size=[0.15, 0.15],
            code_size=8)))

_SEG_META_KEYS = [
    'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
    'ori_lidar2img', 'img_aug_matrix', 'lidar_aug_matrix', 'box_type_3d',
    'sample_idx', 'lidar_path', 'img_path', 'transformation_3d_flow',
    'pcd_rotation', 'pcd_scale_factor', 'pcd_trans', 'num_pts_feats',
    'token', 'scene_id', 'frame_id', 'timestamp', 'num_views'
]

train_pipeline = [
    dict(type='BEVLoadMultiViewImageFromFiles', to_float32=True,
         color_type='color', num_views=1, num_ref_frames=0),
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='SeaPerseaClipIntensity'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True,
         with_attr_label=False),
    dict(type='ImageAug3D', final_dim=[256, 352], resize_lim=[0.55, 0.55],
         bot_pct_lim=[0.0, 0.0], rot_lim=[0.0, 0.0], rand_flip=False,
         is_train=True),
    dict(type='BEVFusionGlobalRotScaleTrans', scale_ratio_range=[1, 1],
         rot_range=[0, 0], translation_std=0),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=['ship', 'buoy', 'other']),
    dict(type='PointShuffle'),
    dict(type='Pack3DDetInputs', keys=[
        'points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'],
         meta_keys=_SEG_META_KEYS)
]

test_pipeline = [
    dict(type='BEVLoadMultiViewImageFromFiles', to_float32=True,
         color_type='color', num_views=1, num_ref_frames=0),
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='SeaPerseaClipIntensity'),
    dict(type='ImageAug3D', final_dim=[256, 352], resize_lim=[0.55, 0.55],
         bot_pct_lim=[0.0, 0.0], rot_lim=[0.0, 0.0], rand_flip=False,
         is_train=False),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['points', 'img'],
         meta_keys=_SEG_META_KEYS)
]

train_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='SeaPerseaDataset',
        data_root='',
        ann_file='infos_train.pkl',
        metainfo=dict(_delete_=True, classes=list(classes)),
        modality=input_modality,
        data_prefix=data_prefix,
        box_type_3d='LiDAR',
        default_cam_key=None,
        filter_empty_gt=False,
        test_mode=False,
        pipeline=train_pipeline))

val_dataloader = None
test_dataloader = None
val_cfg = None
test_cfg = None
val_evaluator = None
test_evaluator = None

metainfo = dict(_delete_=True, classes=list(classes))
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))
auto_scale_lr = dict(enable=False, base_batch_size=32)
param_scheduler = []
train_cfg = dict(by_epoch=True, max_epochs=6, val_interval=1000000)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1),
    checkpoint=dict(type='CheckpointHook', interval=2, by_epoch=True,
                    save_last=True))
log_processor = dict(window_size=1, by_epoch=True)
custom_hooks = []
