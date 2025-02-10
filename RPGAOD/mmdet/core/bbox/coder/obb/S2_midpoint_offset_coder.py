import numpy as np
import torch
import math
from ..base_bbox_coder import BaseBBoxCoder
from mmdet.core.bbox.transforms_obb import obb2hbb, obb2poly, rectpoly2obb
from mmdet.core.bbox.builder import BBOX_CODERS
from mmdet.core.bbox.transforms_obb import regular_theta, regular_obb
pi = math.pi
@BBOX_CODERS.register_module()
class S2MidpointOffsetCoder(BaseBBoxCoder):

    def __init__(self,
                 target_means=(0., 0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1., 1.)):
        super(BaseBBoxCoder, self).__init__()
        self.means = target_means
        self.stds = target_stds

    def encode(self, bboxes, gt_bboxes):
        assert bboxes.size(0) == gt_bboxes.size(0)
        encoded_bboxes = bbox2delta_sp(bboxes, gt_bboxes, self.means, self.stds)
        return encoded_bboxes

    def decode(self,
               bboxes,
               pred_bboxes,
               max_shape=None,
               wh_ratio_clip=16 / 1000):
        assert pred_bboxes.size(0) == bboxes.size(0)
        decoded_bboxes = delta_sp2bbox(bboxes, pred_bboxes, self.means, self.stds,
                                       wh_ratio_clip)
        return decoded_bboxes


def bbox2delta_sp(proposals, gt,
                  means=(0., 0., 0., 0., 0.),
                  stds=(1., 1., 1., 1., 1.)):
    assert proposals.size(0) == gt.size(0)

    proposals = proposals.float()
    gt = gt.float()

    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0]
    ph = proposals[..., 3] - proposals[..., 1]

    
    gx = gt[..., 0]
    gy = gt[..., 1]
    
    gw = gt[..., 2]
    gh = gt[..., 3]
    gtheta = gt[..., 4]
    ghw = gh*torch.abs(torch.sin(gtheta)) +gw*torch.abs(torch.cos(gtheta))
    ghh = gw*torch.abs(torch.sin(gtheta)) +gh*torch.abs(torch.cos(gtheta))
    
    da_1 = 0.5*(torch.abs(gh*torch.sin(gtheta))-torch.abs(gw*torch.cos(gtheta)))/(torch.abs(gh*torch.sin(gtheta))+torch.abs(gw*torch.cos(gtheta)))
    db_1 = 0.5*(torch.abs(gw*torch.sin(gtheta))-torch.abs(gh*torch.cos(gtheta)))/(torch.abs(gw*torch.sin(gtheta))+torch.abs(gh*torch.cos(gtheta)))
    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(ghw/ pw)
    dh = torch.log(ghh / ph)
    da = torch.where(gtheta>=0,da_1,-da_1)

    db = torch.where(gtheta>=0,db_1,-db_1)
    da = torch.where(torch.abs(db)!=0.5,da,dh)
    deltas = torch.stack([dx,dy, dw, da, db], dim=-1)
    
    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


def delta_sp2bbox(rois, deltas,
                  means=(0., 0., 0., 0., 0.),
                  stds=(1., 1., 1., 1., 1.),
                  wh_ratio_clip=16 / 1000):
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 5)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 5)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::5]
    dy = denorm_deltas[:, 1::5]
    dw = denorm_deltas[:, 2::5]
    da = denorm_deltas[:, 3::5]
    db = denorm_deltas[:, 4::5]
    
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    #dh = dh.clamp(min=-max_ratio, max=max_ratio)
    # Compute center of each roi
    px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
    # Compute width/height of each roi
    pw = (rois[:, 2] - rois[:, 0]).unsqueeze(1).expand_as(dw)
    ph = (rois[:, 3] - rois[:, 1]).unsqueeze(1).expand_as(dw)
   
    # Use exp(network energy) to enlarge/shrink each roi
    
    ghw = pw * dw.exp()
    
    # Use network energy to shift the center of each roi
    gx = px + pw * dx
    gy = py + ph * dy 
   
    #da_1 = da.clamp(min=-0.5, max=0.5)
    #da_2 = 2*da
    db = db.clamp(min=-0.5, max=0.5)
    da = torch.where(torch.abs(db)!=0.5,da.clamp(min=-0.5, max=0.5),\
                     da.clamp(min=-max_ratio, max=max_ratio))
    
    ghh = torch.where(torch.abs(db)!=0.5,
                     ghw*torch.sqrt(torch.abs((1-4*torch.square(da))/
                                        (1-4*torch.square(db)) )),ph*da.exp())
    
    gda = torch.where(torch.abs(db)!=0.5,da,db)
    gdb = db
    
    gtheta = torch.where(torch.abs(gdb)!=0.5,        
             torch.atan(torch.sqrt(torch.abs( (1+2*gda)*(1+2*gdb)/(1-2*gda)/(1-2*gdb) ) )),
             0.25*math.pi*(1+2*gdb))
    gw = torch.sqrt(torch.square((0.5-gda)*ghw)+torch.square((0.5+gdb)*ghh))
    gh = torch.sqrt(torch.square((0.5+gda)*ghw)+torch.square((0.5-gdb)*ghh))
    gw_regular = torch.where(gw >= gh, gw, gh)
    gh_regular = torch.where(gw >= gh, gh, gw)
    gtheta_regular = torch.where(gw >= gh, gtheta, gtheta-(gtheta>0)*pi/2+(gtheta<=0)*pi/2)
    obboxes = torch.cat([gx, gy, gw_regular, gh_regular, gtheta_regular], dim=1)
    
    return obboxes


