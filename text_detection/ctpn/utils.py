#-*- coding:utf-8 -*-
import numpy as np
import cv2
from text_detection.ctpn.config import *


# base_anchor anchors reshape [10*h*w, 4]
# [10, h*w, 4] -> [10*h*w, 4]
# feature_map anchor：[10, h*w, 4] -> [h*w, 10, 4] -> [10*h*w, 4]
def gen_anchor( featuresize, scale, 
                heights = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283], 
                widths = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16]):
    h, w = featuresize
    shift_x = np.arange(0, w) * scale
    shift_y = np.arange(0, h) * scale
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel()), axis=1)

    #base center(x,,y) -> (x1, y1, x2, y2)
    base_anchor = np.array([0, 0, 15, 15])
    xt = (base_anchor[0] + base_anchor[2]) * 0.5
    yt = (base_anchor[1] + base_anchor[3]) * 0.5
    heights = np.array(heights).reshape(len(heights), 1)
    widths = np.array(widths).reshape(len(widths), 1)
    x1 = xt - widths * 0.5
    y1 = yt - heights * 0.5
    x2 = xt + widths * 0.5
    y2 = yt + heights * 0.5
    base_anchor = np.hstack((x1, y1, x2, y2))
    
    anchor = list()
    for i in range(base_anchor.shape[0]):
        anchor_x1 = shift[:,0] + base_anchor[i][0]
        anchor_y1 = shift[:,1] + base_anchor[i][1]
        anchor_x2 = shift[:,2] + base_anchor[i][2]
        anchor_y2 = shift[:,3] + base_anchor[i][3]
        anchor.append(np.dstack((anchor_x1, anchor_y1, anchor_x2, anchor_y2)))

    return np.squeeze(np.array(anchor)).transpose((1,0,2)).reshape((-1, 4))


# anchor bbox iou
# iou = inter_area/(bb_area + anchor_area - inter_area)
def compute_iou(anchors, bbox):
    ious = np.zeros((len(anchors), len(bbox)), dtype=np.float32)
    anchor_area = (anchors[:,2] - anchors[:,0])*(anchors[:,3] - anchors[:,1])
    for num, _bbox in enumerate(bbox):
        bb = np.tile(_bbox,(len(anchors), 1))
        bb_area = (bb[:,2] - bb[:,0])*(bb[:,3] - bb[:,1])
        inter_h = np.maximum(np.minimum(bb[:,3], anchors[:,3]) - np.maximum(bb[:,1], anchors[:,1]), 0)
        inter_w = np.maximum(np.minimum(bb[:,2], anchors[:,2]) - np.maximum(bb[:,0], anchors[:,0]), 0)
        inter_area = inter_h*inter_w
        ious[:,num] = inter_area/(bb_area + anchor_area - inter_area)

    return ious


# anchor gtboxes regression_factor(Vc, Vh)
# 1. (x1, y1, x2, y2) -> (ctr_x, ctr_y, w, h)
# 2. Vc = (gt_y - anchor_y) / anchor_h
#     Vh = np.log(gt_h / anchor_h)
def bbox_transfrom(anchors, gtboxes):
    gt_y = (gtboxes[:, 1] + gtboxes[:, 3]) * 0.5
    gt_h = gtboxes[:, 3] - gtboxes[:, 1] + 1.0

    anchor_y = (anchors[:, 1] + anchors[:, 3]) * 0.5
    anchor_h = anchors[:, 3] - anchors[:, 1] + 1.0

    Vc = (gt_y - anchor_y) / anchor_h
    Vh = np.log(gt_h / anchor_h)

    return np.vstack((Vc, Vh)).transpose()


# anchor regression_factor(Vc, Vh) bbox
def transform_bbox(anchor, regression_factor):
    anchor_y = (anchor[:, 1] + anchor[:, 3]) * 0.5
    anchor_x = (anchor[:, 0] + anchor[:, 2]) * 0.5
    anchor_h = anchor[:, 3] - anchor[:, 1] + 1

    Vc = regression_factor[0, :, 0]
    Vh = regression_factor[0, :, 1]

    bbox_y = Vc * anchor_h + anchor_y
    bbox_h = np.exp(Vh) * anchor_h

    x1 = anchor_x - 16 * 0.5
    y1 = bbox_y - bbox_h * 0.5
    x2 = anchor_x + 16 * 0.5
    y2 = bbox_y + bbox_h * 0.5
    bbox = np.vstack((x1, y1, x2, y2)).transpose()

    return bbox


# bbox
#     x1 >= 0
#     y1 >= 0
#     x2 < im_shape[1]
#     y2 < im_shape[0]
def clip_bbox(bbox, im_shape):
    bbox[:, 0] = np.maximum(np.minimum(bbox[:, 0], im_shape[1] - 1), 0)
    bbox[:, 1] = np.maximum(np.minimum(bbox[:, 1], im_shape[0] - 1), 0)
    bbox[:, 2] = np.maximum(np.minimum(bbox[:, 2], im_shape[1] - 1), 0)
    bbox[:, 3] = np.maximum(np.minimum(bbox[:, 3], im_shape[0] - 1), 0)

    return bbox


def filter_bbox(bbox, minsize):
    ws = bbox[:, 2] - bbox[:, 0] + 1
    hs = bbox[:, 3] - bbox[:, 1] + 1
    keep = np.where((ws >= minsize) & (hs >= minsize))[0]
    return keep


# RPN module
# 1. anchor
# 2. anchor gtboxes iou
# 3. iou，anchor，0，1，-1
#     (1) bbox, iou anchor
#     (2) anchor，bbox iou max_overlap
#     (3) max_overlap anchor
# 4. anchor -1
# 5. anchor max_overlap gtbbox(Vc, Vh)
def cal_rpn(imgsize, featuresize, scale, gtboxes):
    base_anchor = gen_anchor(featuresize, scale)
    overlaps = compute_iou(base_anchor, gtboxes)

    gt_argmax_overlaps = overlaps.argmax(axis=0)
    anchor_argmax_overlaps = overlaps.argmax(axis=1)
    anchor_max_overlaps = overlaps[range(overlaps.shape[0]), anchor_argmax_overlaps]

    labels = np.empty(base_anchor.shape[0])
    labels.fill(-1)
    labels[gt_argmax_overlaps] = 1
    labels[anchor_max_overlaps > IOU_POSITIVE] = 1
    labels[anchor_max_overlaps < IOU_NEGATIVE] = 0

    outside_anchor = np.where(
        (base_anchor[:, 0] < 0) |
        (base_anchor[:, 1] < 0) |
        (base_anchor[:, 2] >= imgsize[1]) |
        (base_anchor[:, 3] >= imgsize[0])
    )[0]
    labels[outside_anchor] = -1

    fg_index = np.where(labels == 1)[0]
    if (len(fg_index) > RPN_POSITIVE_NUM):
        labels[np.random.choice(fg_index, len(fg_index) - RPN_POSITIVE_NUM, replace=False)] = -1
    if not OHEM:
        bg_index = np.where(labels == 0)[0]
        num_bg = RPN_TOTAL_NUM - np.sum(labels == 1)
        if (len(bg_index) > num_bg):
            labels[np.random.choice(bg_index, len(bg_index) - num_bg, replace=False)] = -1

    bbox_targets = bbox_transfrom(base_anchor, gtboxes[anchor_argmax_overlaps, :])

    return [labels, bbox_targets]


def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


class Graph:
    def __init__(self, graph):
        self.graph = graph

    def sub_graphs_connected(self):
        sub_graphs = []
        for index in range(self.graph.shape[0]):
            if not self.graph[:, index].any() and self.graph[index, :].any():
                v = index
                sub_graphs.append([v])
                while self.graph[v, :].any():
                    v = np.where(self.graph[v, :])[0][0]
                    sub_graphs[-1].append(v)

        return sub_graphs


class TextLineCfg:
    SCALE = 600
    MAX_SCALE = 1200
    TEXT_PROPOSALS_WIDTH = 16
    MIN_NUM_PROPOSALS = 2
    MIN_RATIO = 0.5
    LINE_MIN_SCORE = 0.9
    TEXT_PROPOSALS_MIN_SCORE = 0.7
    TEXT_PROPOSALS_NMS_THRESH = 0.3
    MAX_HORIZONTAL_GAP = 60
    MIN_V_OVERLAPS = 0.6
    MIN_SIZE_SIM = 0.6


class TextProposalGraphBuilder:
    def get_successions(self, index):
        box = self.text_proposals[index]
        results = []
        for left in range(int(box[0]) + 1, min(int(box[0]) + TextLineCfg.MAX_HORIZONTAL_GAP + 1, self.im_size[1])):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results

        return results

    def get_precursors(self, index):
        box = self.text_proposals[index]
        results = []
        for left in range(int(box[0]) - 1, max(int(box[0] - TextLineCfg.MAX_HORIZONTAL_GAP), 0) - 1, -1):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results

        return results

    def is_succession_node(self, index, succession_index):
        precursors = self.get_precursors(succession_index)
        if self.scores[index] >= np.max(self.scores[precursors]):
            return True

        return False

    def meet_v_iou(self, index1, index2):
        # overlaps_v: iou_v = inv_y/min(h1, h2)
        # size_similarity: sim = min(h1, h2)/max(h1, h2)
        def overlaps_v(index1, index2):
            h1 = self.heights[index1]
            h2 = self.heights[index2]
            y0 = max(self.text_proposals[index2][1], self.text_proposals[index1][1])
            y1 = min(self.text_proposals[index2][3], self.text_proposals[index1][3])
            return max(0, y1 - y0 + 1) / min(h1, h2)

        def size_similarity(index1, index2):
            h1 = self.heights[index1]
            h2 = self.heights[index2]
            return min(h1, h2) / max(h1, h2)

        return overlaps_v(index1, index2) >= TextLineCfg.MIN_V_OVERLAPS and \
                size_similarity(index1, index2) >= TextLineCfg.MIN_SIZE_SIM

    def build_graph(self, text_proposals, scores, im_size):
        self.text_proposals = text_proposals
        self.scores = scores
        self.im_size = im_size
        self.heights = text_proposals[:, 3] - text_proposals[:, 1] + 1

        boxes_table = [[] for _ in range(self.im_size[1])]
        for index, box in enumerate(text_proposals):
            boxes_table[int(box[0])].append(index)
        self.boxes_table = boxes_table

        graph = np.zeros((text_proposals.shape[0], text_proposals.shape[0]), np.bool)

        for index, box in enumerate(text_proposals):
            successions = self.get_successions(index)
            if len(successions) == 0:
                continue
            succession_index = successions[np.argmax(scores[successions])]
            if self.is_succession_node(index, succession_index):
                graph[index, succession_index] = True

        return Graph(graph)


class TextProposalConnectorOriented:
    def __init__(self):
        self.graph_builder = TextProposalGraphBuilder()

    def group_text_proposals(self, text_proposals, scores, im_size):
        graph = self.graph_builder.build_graph(text_proposals, scores, im_size)
        
        return graph.sub_graphs_connected()

    def fit_y(self, X, Y, x1, x2):
        if np.sum(X == X[0]) == len(X):
            return Y[0], Y[0]
        p = np.poly1d(np.polyfit(X, Y, 1))
        return p(x1), p(x2)

    def get_text_lines(self, text_proposals, scores, im_size):
        tp_groups = self.group_text_proposals(text_proposals, scores, im_size) 
        
        text_lines = np.zeros((len(tp_groups), 8), np.float32)
        for index, tp_indices in enumerate(tp_groups):
            text_line_boxes = text_proposals[list(tp_indices)]

            X = (text_line_boxes[:, 0] + text_line_boxes[:, 2]) / 2
            Y = (text_line_boxes[:, 1] + text_line_boxes[:, 3]) / 2
            x0 = np.min(text_line_boxes[:, 0])
            x1 = np.max(text_line_boxes[:, 2])

            z1 = np.polyfit(X, Y, 1) 

            offset = (text_line_boxes[0, 2] - text_line_boxes[0, 0]) * 0.5 

            lt_y, rt_y = self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 1], x0 + offset, x1 - offset)
            lb_y, rb_y = self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 3], x0 + offset, x1 - offset)

            score = scores[list(tp_indices)].sum() / float(len(tp_indices))

            text_lines[index, 0] = x0
            text_lines[index, 1] = min(lt_y, rt_y)
            text_lines[index, 2] = x1
            text_lines[index, 3] = max(lb_y, rb_y)
            text_lines[index, 4] = score
            text_lines[index, 5] = z1[0]
            text_lines[index, 6] = z1[1]
            height = np.mean((text_line_boxes[:, 3] - text_line_boxes[:, 1]))
            text_lines[index, 7] = height + 2.5

        text_recs = np.zeros((len(text_lines), 9), np.float)
        index = 0
        for line in text_lines:
            b1 = line[6] - line[7] / 2 
            b2 = line[6] + line[7] / 2
            x1 = line[0]
            y1 = line[5] * line[0] + b1
            x2 = line[2]
            y2 = line[5] * line[2] + b1
            x3 = line[0]
            y3 = line[5] * line[0] + b2
            x4 = line[2]
            y4 = line[5] * line[2] + b2
            disX = x2 - x1
            disY = y2 - y1
            width = np.sqrt(disX * disX + disY * disY)

            fTmp0 = y3 - y1
            fTmp1 = fTmp0 * disY / width
            x = np.fabs(fTmp1 * disX / width)
            y = np.fabs(fTmp1 * disY / width)
            if line[5] < 0:
                x1 -= x
                y1 += y
                x4 += x
                y4 -= y
            else:
                x2 += x
                y2 += y
                x3 -= x
                y3 -= y
            text_recs[index, 0] = x1
            text_recs[index, 1] = y1
            text_recs[index, 2] = x2
            text_recs[index, 3] = y2
            text_recs[index, 4] = x3
            text_recs[index, 5] = y3
            text_recs[index, 6] = x4
            text_recs[index, 7] = y4
            text_recs[index, 8] = line[4]
            index = index + 1

        return text_recs

if __name__=='__main__':
    anchor = gen_anchor((10, 15), 16)
