_base_ = './s2anet_r50_fpn_3x_hrsc.py'

# model
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
