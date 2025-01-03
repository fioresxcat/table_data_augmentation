import xml.etree.ElementTree as ET
import os
import pdb
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import json
import shutil


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
    obj_names = [obj_names[i] for i in indices]
    return boxes, obj_names



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


def extract_cells(rows, cols, spans):
    cells = []
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            xr1, yr1, xr2, yr2 = row
            xc1, yc1, xc2, yc2 = col
            # cells.append({'bbox':[xc1, yr1, xc2, yr2], 'relation':[i, i+1, j, j+1]})
            # Now relation of a cell correspond to row and col
            cells.append({'bbox':[xc1, yr1, xc2, yr2], 'relation':[i, i, j, j]})
        
    ## Replace span cell into cells
    '''
    Idea: Xét 1 cell
        - nếu cell này ko thuộc về span cell nào -> Lấy
        - nếu cell này thuộc về 1 span cell:
            + nếu chưa lấy span cell của cell này 
                --> Lấy span cell, cho relative của span cell chính là cell đang xét
            + nếu đã lấy spann cell của cell này 
                --> Tăng relative của span cell lên
    '''

    if len(spans) > 0: 
        new_cells = []
        flags = {str(span):False for span in spans}
        for i, cell in enumerate(cells):
            span_of_cell = get_span_of_cell(cell['bbox'], spans)
            if span_of_cell is None:
                new_cells.append(cell)
                continue
            if not flags[str(span_of_cell)]:
                new_cells.append({'bbox':span_of_cell, 'relation':cell['relation']})
                flags[str(span_of_cell)] = True
            else:
                idx = [k for k, cell in enumerate(new_cells) if str(cell['bbox'])==str(span_of_cell)][0]
                sr = min(new_cells[idx]['relation'][0], cell['relation'][0])
                er = max(new_cells[idx]['relation'][1], cell['relation'][1])
                sc = min(new_cells[idx]['relation'][2], cell['relation'][2])
                ec = max(new_cells[idx]['relation'][3], cell['relation'][3])
                new_cells[idx]['relation'] = [sr, er, sc, ec]
    else:
        new_cells = cells
    
    return new_cells