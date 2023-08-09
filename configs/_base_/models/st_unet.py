# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='TransResNetV2',
        # depth=18,
        # num_stages=4,
        # out_indices=(1, 2, 3, 4),
        # dilations=(1, 1, 2, 4),
        # strides=(1, 2, 1, 1),
        # norm_cfg=norm_cfg,
        # norm_eval=False,
        # style='pytorch',
        # contract_dilation=True
        ),
    decode_head=dict(
        type='STUnetHead',
        # in_channels=(64, 128, 256, 512),
        # channels=64,
        dropout_ratio=0.1,
        # window_size=8,
        num_classes=6,
        in_index=(0, 1, 2, 3, 4),
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
