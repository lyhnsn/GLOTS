# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='SwinUnet',
        img_size=224,
        embed_dim=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        # depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        # strides=(4, 2, 2, 2),
        # out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3),
        # use_abs_pos_embed=False,
        # act_cfg=dict(type='GELU'),
        # norm_cfg=backbone_norm_cfg),
    decode_head=dict(
        type='FCNHead',
        in_channels=96,
        in_index=0,
        channels=96,
        num_convs=0,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=norm_cfg,
        align_corners=False,
        # input_transform='resize_concat',
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # auxiliary_head=dict(
    #     type='FCNHead',
    #     in_channels=128,
    #     in_index=3,
    #     channels=64,
    #     num_convs=1,
    #     concat_input=False,
    #     dropout_ratio=0.1,
    #     num_classes=2,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
