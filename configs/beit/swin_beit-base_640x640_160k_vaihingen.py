_base_ = [
    '../_base_/models/swin_beit.py', '../_base_/datasets/vaihingen_640.py',
    '../_base_/default_runtime_vis.py', '../_base_/schedules/schedule_160k.py'
]

# model cfg
model = dict(
    # pretrained='pretrain/beit_base_patch16_224_pt22k_ft22k.pth',
    pretrained='pretrain/beitv2_base_patch16_224_pt1k_ft21k.pth',
    decode_head=dict(num_classes=6), # reversely index
    auxiliary_head=dict(num_classes=6),
    test_cfg=dict(crop_size=(640, 640)))

# optimizer cfg
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=3e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2560, 640),
        img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True, min_size=640),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    # val=dict(pipeline=test_pipeline),
    # test=dict(pipeline=test_pipeline),
    samples_per_gpu=1)
