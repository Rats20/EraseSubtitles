#-*- coding:utf-8 -*-
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from text_detection.ctpn import config
from text_detection.ctpn.ctpn import CTPN_Model
from text_detection.ctpn.utils import gen_anchor, transform_bbox, clip_bbox, filter_bbox, nms, TextProposalConnectorOriented


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights = 'text_detection/weights/new_ctpn_ep09_0.0420_0.0198_0.0618.pth'
model = CTPN_Model().to(device)
model.load_state_dict(torch.load(weights, map_location=device)['model_state_dict'])
model.eval()



def get_text_boxes(image, display = False, prob_thresh = 0.5):
    h, w= image.shape[:2]
    rescale_fac = max(h, w) / 1000
    if rescale_fac > 1.0:
        h = int(h / rescale_fac)
        w = int(w / rescale_fac)
        image = cv2.resize(image, (w,h))
        h, w = image.shape[:2]
    image = image.astype(np.float32) - config.IMAGE_MEAN
    image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        cls, regr = model(image)
        cls_prob = F.softmax(cls, dim=-1).cpu().numpy()
        regr = regr.cpu().numpy()
        anchor = gen_anchor((int(h / 16), int(w / 16)), 16)
        bbox = transform_bbox(anchor, regr)
        bbox = clip_bbox(bbox, [h, w])

        fg = np.where(cls_prob[0, :, 1] > prob_thresh)[0]
        select_anchor = bbox[fg, :]
        select_score = cls_prob[0, fg, 1]
        select_anchor = select_anchor.astype(np.int32)
        keep_index = filter_bbox(select_anchor, 16)

        select_anchor = select_anchor[keep_index]
        select_score = select_score[keep_index]
        select_score = np.reshape(select_score, (select_score.shape[0], 1))
        nmsbox = np.hstack((select_anchor, select_score))
        keep = nms(nmsbox, 0.3)
        select_anchor = select_anchor[keep]
        select_score = select_score[keep]

        textConn = TextProposalConnectorOriented()
        text = textConn.get_text_lines(select_anchor, select_score, [h, w])
        
        return text

    
