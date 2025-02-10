import BboxToolkit as bt
import mmcv
import numpy as np

from mmdet.core import arb_mapping, tensor2imgs
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
from .obb_base import OBBBaseDetector
from .obb_test_mixins import RotateAugRPNTestMixin


@DETECTORS.register_module()
class OBBRPN(OBBBaseDetector, RotateAugRPNTestMixin):
    """Implementation of Oriented Region Proposal Network"""

    def __init__(self,
                 backbone,
                 neck,
                 rpn_head,
                 train_cfg,
                 test_cfg,
                 pretrained=None):
        super(OBBRPN, self).__init__()
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck) if neck is not None else None
        rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
        rpn_head.update(train_cfg=rpn_train_cfg)
        rpn_head.update(test_cfg=test_cfg.rpn)
        self.rpn_head = build_head(rpn_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(OBBRPN, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            self.neck.init_weights()
        self.rpn_head.init_weights()

    def extract_feat(self, img):
        """Extract features

        Args:
            img (torch.Tensor): Image tensor with shape (n, c, h ,w).

        Returns:
            list[torch.Tensor]: Multi-level features that may have
                different resolutions.
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Dummy forward function"""
        x = self.extract_feat(img)
        rpn_outs = self.rpn_head(x)
        return rpn_outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_obboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_obboxes_ignore=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        if self.train_cfg.rpn.get('debug', False):
            self.rpn_head.debug_imgs = tensor2imgs(img)

        x = self.extract_feat(img)

        proposal_type = getattr(self.rpn_head, 'bbox_type', 'hbb')
        target_bboxes = gt_bboxes if proposal_type == 'hbb' else gt_obboxes
        target_bboxes_ignore = gt_bboxes_ignore if proposal_type == 'hbb' \
                else gt_obboxes_ignore
        losses = self.rpn_head.forward_train(x, img_metas, target_bboxes, None,
                                             target_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            np.ndarray: proposals
        """
        x = self.extract_feat(img)
        outs = self.rpn_head(x)
        bbox_list = self.rpn_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        bbox_type = getattr(self.rpn_head, 'bbox_type', 'hbb')
        bbox_results = [
            arb2result(det_bboxes, det_labels, self.rpn_head.num_classes, bbox_type)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation"""
        raise NotImplementedError

    def show_result(self,
                    img,
                    result,
                    colors='green',
                    top_k=300,
                    thickness=1,
                    win_name='',
                    wait_time=0,
                    show=False,
                    out_file=None,
                    score_thr=None):
        img = mmcv.imread(img)
        bboxes, scores = result[:, :-1], result[:, -1]
        idx = scores.argsort()[::-1]
        bboxes = bboxes[idx]

        top_k = min(top_k, len(bboxes))
        bboxes = bboxes[:top_k, :]

        if out_file is not None:
            show = False
        img = bt.imshow_bboxes(
            img,
            bboxes,
            colors=colors,
            thickness=thickness,
            with_text=False,
            show=show,
            win_name=win_name,
            wait_time=wait_time,
            out_file=out_file)
        return img
