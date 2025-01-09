import xml.etree.ElementTree as ET
import os
import pdb
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import json
import shutil
from typing_extensions import List, Dict, Tuple, Literal, Optional


def max_left(poly):
    return min(poly[0], poly[2], poly[4], poly[6])

def max_right(poly):
    return max(poly[0], poly[2], poly[4], poly[6])

def row_polys(polys):
    polys.sort(key=lambda x: max_left(x))
    clusters, y_min = [], []
    for tgt_node in polys:
        if len (clusters) == 0:
            clusters.append([tgt_node])
            y_min.append(tgt_node[1])
            continue
        matched = None
        tgt_7_1 = tgt_node[7] - tgt_node[1]
        min_tgt_0_6 = min(tgt_node[0], tgt_node[6])
        max_tgt_2_4 = max(tgt_node[2], tgt_node[4])
        max_left_tgt = max_left(tgt_node)
        for idx, clt in enumerate(clusters):
            src_node = clt[-1]
            src_5_3 = src_node[5] - src_node[3]
            max_src_2_4 = max(src_node[2], src_node[4])
            min_src_0_6 = min(src_node[0], src_node[6])
            overlap_y = (src_5_3 + tgt_7_1) - (max(src_node[5], tgt_node[7]) - min(src_node[3], tgt_node[1]))
            overlap_x = (max_src_2_4 - min_src_0_6) + (max_tgt_2_4 - min_tgt_0_6) - (max(max_src_2_4, max_tgt_2_4) - min(min_src_0_6, min_tgt_0_6))
            if overlap_y > 0.5*min(src_5_3, tgt_7_1) and overlap_x < 0.6*min(max_src_2_4 - min_src_0_6, max_tgt_2_4 - min_tgt_0_6):
                distance = max_left_tgt - max_right(src_node)
                if matched is None or distance < matched[1]:
                    matched = (idx, distance)
        if matched is None:
            clusters.append([tgt_node])
            y_min.append(tgt_node[1])
        else:
            idx = matched[0]
            clusters[idx].append(tgt_node)
    zip_clusters = list(zip(clusters, y_min))
    zip_clusters.sort(key=lambda x: x[1])
    zip_clusters = list(np.array(zip_clusters, dtype=object)[:, 0])
    return zip_clusters


def row_bbs(bbs):
    polys = []
    poly2bb = {}
    for bb in bbs:
        poly = [bb[0], bb[1], bb[2], bb[1], bb[2], bb[3], bb[0], bb[3]]
        polys.append(poly)
        poly2bb[tuple(poly)] = bb
    poly_rows = row_polys(polys)
    bb_rows = []
    for row in poly_rows:
        bb_row = []
        for poly in row:
            bb_row.append(poly2bb[tuple(poly)])
        bb_rows.append(bb_row)
    return bb_rows


def sort_bbs(bbs):
    bb2idx_original = {tuple(bb): i for i, bb in enumerate(bbs)}
    bb_rows = row_bbs(bbs)
    sorted_bbs = [bb for row in bb_rows for bb in row]
    sorted_indices = [bb2idx_original[tuple(bb)] for bb in sorted_bbs]
    return sorted_bbs, sorted_indices


def sort_polys(polys):
    poly2idx_original = {tuple(poly): i for i, poly in enumerate(polys)}
    poly_clusters = row_polys(polys)
    sorted_polys = []
    for row in poly_clusters:
        sorted_polys.extend(row)
    sorted_indices = [poly2idx_original[tuple(poly)] for poly in sorted_polys]
    return polys, sorted_indices



def is_image(fp):
    if isinstance(fp, str):
        fp = Path(fp)
    return fp.suffix in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']


def parse_xml(xml):
    root = ET.parse(xml).getroot()
    objs = root.findall('object')
    boxes, ymins, obj_names = [], [], []
    for obj in objs:
        obj_name = obj.find('name').text
        box = obj.find('bndbox')
        xmin = float(box.find('xmin').text)
        ymin = float(box.find('ymin').text)
        xmax = float(box.find('xmax').text)
        ymax = float(box.find('ymax').text)
        ymins.append(ymin)
        boxes.append([xmin, ymin, xmax, ymax])
        obj_names.append(obj_name)
    indices = np.argsort(ymins)
    boxes = [boxes[i] for i in indices]
    boxes = [list(map(int, bb)) for bb in boxes]
    obj_names = [obj_names[i] for i in indices]
    return boxes, obj_names


def poly2box(poly):
    poly = np.array(poly).flatten().tolist()
    xmin, xmax = min(poly[::2]), max(poly[::2])
    ymin, ymax = min(poly[1::2]), max(poly[1::2])
    return [xmin, ymin, xmax, ymax]


def iou_bbox(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # Compute the area of the intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the Intersection over Union (IoU)
    r1 = interArea / boxAArea
    r2 = interArea / boxBArea
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return r1, r2, iou


def get_span_of_cell(cell, spans):
    for span in spans:
        r1, r2, iou = iou_bbox(cell, span)
        if r1 > 0.5:
            return span
    return None


def iou_axis(start1, end1, start2, end2):
    intersection = max(0, min(end1, end2) - max(start1, start2))
    union = (end1 - start1) + (end2 - start2) - intersection
    iou = intersection / union if union > 0 else 0
    r1 = intersection / abs(end1 - start1)
    r2 = intersection / abs(end2 - start2)
    return r1, r2, iou


def get_bb_type(boxes, names):
    rows, cols, spans = [], [], []
    for box, name in zip(boxes, names):
        if name == 'row':
            rows.append(box)
        elif name == 'col':
            cols.append(box)
        elif name == 'span':
            spans.append(box)
    rows.sort(key=lambda x: x[1])
    cols.sort(key=lambda x: x[0])
    spans.sort(key=lambda x: x[1])
    return rows, cols, spans


def is_row_valid(row, spans, overlap_threshold=0):
    for span in spans:
        if not is_box_is_span(row, span):
            r1, r2, iou = iou_axis(row[1], row[3], span[1], span[3])
            if r1 > overlap_threshold:
                return False
    return True


def get_bb_size(bb):
    return (bb[2]-bb[0], bb[3]-bb[1])


def is_line_black(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold = 128
    black_pixels = np.sum(image < threshold)
    white_pixels = np.sum(image >= threshold)
    return black_pixels > 0.1 * (black_pixels+white_pixels)


def is_box_is_span(box, span):
    return all([abs(box[i]-span[i]) < 5 for i in range(len(box))])


def mask_image(im: np.ndarray, bbox, color: tuple):
    xmin, ymin, xmax, ymax = bbox
    im[ymin:ymax, xmin:xmax] = color
    return im


def is_span_cell(cell):
    rel = cell['relation']
    return rel[1] - rel[0] > 0 or rel[3] - rel[2] > 0