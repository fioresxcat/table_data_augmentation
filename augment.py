from utils import *


class BaseAugmenter:
    def __init__(self):
        pass
    
        


class MaskLineAugmenter(BaseAugmenter):
    def __init__(self):
        super().__init__()
        self.line_thickness = 5


    def get_nearby_region(self, bbox, axis='vertical'):
        if axis == 'vertical':
            if bbox[0] >= self.line_thickness:
                xmin = bbox[0] - self.line_thickness
                xmax = bbox[0]
            else:
                xmin = bbox[2]
                xmax = bbox[2] + self.line_thickness
            ymin = bbox[1]
            ymax = bbox[3]
        elif axis == 'horizontal':
            if bbox[1] >= self.line_thickness:
                ymin = bbox[1] - self.line_thickness
                ymax = bbox[1]
            else:
                ymin = bbox[3]
                ymax = bbox[3] + self.line_thickness
            xmin = bbox[0]
            xmax = bbox[2]
        return xmin, ymin, xmax, ymax
    
    
    def process(self, im: Image, bbs: list):
        im = np.array(im)[:, :, ::-1]
        orig_im = im.copy()
        bbs = [list(map(int, bb)) for bb in bbs]
        # Iterate through all bounding boxes
        for i, bbox1 in enumerate(bbs):
            x1_min, y1_min, x1_max, y1_max = bbox1
            for j, bbox2 in enumerate(bbs):
                if i >= j:  # Avoid redundant comparisons
                    continue
                x2_min, y2_min, x2_max, y2_max = bbox2
                # Check for vertical shared border
                if x1_max == x2_min or x2_max == x1_min and (min(y1_max, y2_max) - max(y1_min, y2_min) >= 0):
                    x_start = min(x1_max, x2_min)
                    x_end = x_start + self.line_thickness
                    y_start = max(y1_min, y2_min)
                    y_end = min(y1_max, y2_max)
                    nearby_bb = self.get_nearby_region(bbox=[x_start, y_start, x_end, y_end], axis='vertical')
                    nearby_region = orig_im[nearby_bb[1]:nearby_bb[3], nearby_bb[0]:nearby_bb[2]]
                    im[y_start:y_end, x_start:x_end] = nearby_region
                    # cv2.imwrite('test.png', im)
                    # pdb.set_trace()


                # Check for horizontal shared border
                if y1_max == y2_min or y2_max == y1_min:
                    y_start = min(y1_max, y2_min)
                    y_end = y_start + self.line_thickness
                    x_start = max(x1_min, x2_min)
                    x_end = min(x1_max, x2_max)
                    nearby_bb = self.get_nearby_region(bbox=[x_start, y_start, x_end, y_end], axis='horizontal')
                    nearby_region = orig_im[nearby_bb[1]:nearby_bb[3], nearby_bb[0]:nearby_bb[2]]
                    im[y_start:y_end, x_start:x_end] = nearby_region

        return Image.fromarray(im[:, :, ::-1])
    


if __name__ == "__main__":
    pass

    augmenter = MaskLineAugmenter()
    dir = 'temp_pdf_imgs'
    for ip in Path(dir).glob('*'):
        if '0._bao_cao_chinh_158_table_0' not in ip.stem:
            continue
        if not is_image(ip):
            continue
        xp = ip.with_suffix('.xml')
        if not xp.exists():
            continue
        im = Image.open(ip)
        boxes, names = parse_xml(xp)
        rows, cols, spans = [], [], []
        for bb, name in zip(boxes, names):
            if name == 'row':
                rows.append(bb)
            elif name == 'col':
                cols.append(bb)
            elif name == 'span':
                spans.append(bb)
        cells = extract_cells(rows, cols, spans)
        cell_bbs = [cell['bbox'] for cell in cells]
        im_aug = augmenter.process(im, cell_bbs)
        im_aug.save('test.png')