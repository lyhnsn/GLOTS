# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='pretrain/deit_base_distilled_patch16_384.pth',
    backbone=dict(
        type='DistilledVisionTransformer',
        img_size=384,
        embed_dim=768,
        patch_size=16,
        mlp_ratio=4,
        depth=12,
        num_heads=12,
        out_indices=(3, 5, 7, 11),
        qkv_bias=True,
        # qk_scale=None,
        # patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        norm_cfg=dict(type='LN', eps=1e-6),
        # act_cfg=dict(type='GELU'),
        # norm_eval=False,
        init_values=0.1),
    neck=dict(type='Feature2Pyramid', embed_dim=768, rescales=[4, 2, 1, 0.5]),
    decode_head=dict(
        type='SwinHead',
        # in_channels=3,
        embed_dims=768, # TODO start: model args
        # depths=[2, 2, 2, 2], # TODO: out of memory
        depths=[1, 1, 3, 1],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        drop_path_rate=0.1,
        use_abs_pos_embed=False,
        patch_norm=True,
        mlp_ratio=4, # TODO end
        in_channels=[768, 768, 768, 768], # TODO start: head args
        # in_index=[3, 2, 1, 0], # reverse order
        in_index=[0, 1, 2, 3], # normal order
        channels=768,
        dropout_ratio=0.1,
        num_classes=150,
        # norm_cfg=norm_cfg,
        act_cfg=dict(type='GELU'),
        align_corners=False, # TODO end
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
