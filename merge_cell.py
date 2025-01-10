from utils import *
from base import BaseAugmenter
from mask_line import get_background_color, get_nearby_line_bb

# np.random.seed(58)


class MergeCellAugmenter(BaseAugmenter):
    def __init__(self):
        super().__init__()
        self.max_merge_horizontal = 5 # not include anchor cell yet -> max 6 cells
        self.max_merge_vertical = 5 # not include anchor cell yet -> max 6 cells
        self.line_thickness = 3
    

    # def remove_line_by_cells(self, im, merge_cells, start_row_idx, last_row_idx, last_col_idx):
    #     """
    #         remove all inside borders by removing the right-vertical edge and bottom-horizontal edge of each cell
    #     """
    #     for row_idx in range(start_row_idx, last_row_idx+1):
    #         cells = [cell for cell in merge_cells if cell['relation'][0] == row_idx]
    #         cells.sort(key=lambda cell: cell['relation'][2])
    #         for cell in cells:
    #             rel = cell['relation']
    #             bb = cell['bbox']
    #             xmin, ymin, xmax, ymax = bb
    #             if rel[1] != last_row_idx: # mask below edge
    #                 line_ymin = max(0, ymax-self.line_thickness//2)
    #                 line_ymax = line_ymin + self.line_thickness
    #                 line_bb = [xmin, line_ymin, xmax, line_ymax]
    #                 nearby_bb = get_nearby_line_bb(line_bb, axis='horizontal', line_thickness=self.line_thickness)
    #                 nearby_region = im[nearby_bb[1]:nearby_bb[3], nearby_bb[0]:nearby_bb[2]]
    #                 im[line_bb[1]:line_bb[3], line_bb[0]:line_bb[2]] = nearby_region
    #             if rel[3] != last_col_idx: # mask right edge
    #                 line_xmin = max(0, xmax-self.line_thickness//2)
    #                 line_xmax = line_xmin + self.line_thickness
    #                 line_bb = [line_xmin, ymin, line_xmax, ymax]
    #                 nearby_bb = get_nearby_line_bb(line_bb, axis='vertical', line_thickness=self.line_thickness)
    #                 nearby_region = im[nearby_bb[1]:nearby_bb[3], nearby_bb[0]:nearby_bb[2]]
    #                 im[line_bb[1]:line_bb[3], line_bb[0]:line_bb[2]] = nearby_region


    def narrow_cell_bbox(self, text_bb, cell_bbox):
        text_bb[0] = max(text_bb[0], cell_bbox[0]+self.line_thickness//2)
        text_bb[1] = max(text_bb[1], cell_bbox[1]+self.line_thickness//2)
        text_bb[2] = min(text_bb[2], cell_bbox[2]-self.line_thickness//2)
        text_bb[3] = min(text_bb[3], cell_bbox[3]-self.line_thickness//2)
        return text_bb
    

    def check(self, im: Image, rows, cols, spans, texts):
        """
            check if the image is valid for this augmentation
            condition:
             + must have black border line
        """
        # check if all border line is black
        im = np.array(im)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

        non_overlap_row_indexes = []
        for row_idx, row_bb in enumerate(rows):
            if row_idx == 0:
                continue
            row_bb = list(map(int, row_bb))
            # ---- split by span cells ----
            has_overlap = False
            for span in spans:
                if abs(row_bb[1]-span[1] > 5):  # to skip span trùng với row luôn, như thế thì ko tính là overlap
                    r1, r2, iou = iou_axis(row_bb[1], row_bb[3], span[1], span[3])
                    if r1 > 0.7:
                        has_overlap = True
                        break
            if not has_overlap:
                non_overlap_row_indexes.append(row_idx)

        black_rows, total_rows = 0, 0
        for row_idx in non_overlap_row_indexes:
            row_bb = rows[row_idx]
            ymin = max(0, row_bb[1] - self.line_thickness // 2)
            ymax = ymin + self.line_thickness
            line_bb = [row_bb[0], ymin, row_bb[2], ymax]
            line_image = im[ymin:ymax, row_bb[0]:row_bb[2]]
            if is_image_black(line_image, min_black_percent=0.1):
                black_rows += 1
            total_rows += 1


        non_overlap_col_indexes = []
        for col_idx, col_bb in enumerate(cols):
            if col_idx == 0:
                continue
            col_bb = list(map(int, col_bb))
            # ---- split by span cells ----
            has_overlap = False
            for span in spans:
                if not is_box_is_span(col_bb, span):
                    c1, c2, iou = iou_axis(col_bb[0], col_bb[2], span[0], span[2])
                    if c1 > 0.7:
                        has_overlap = True
            if not has_overlap:
                non_overlap_col_indexes.append(col_idx)
        black_cols, total_cols = 0, 0
        for col_idx in non_overlap_col_indexes:
            col_bb = cols[col_idx]
            xmin = max(0, col_bb[0] - self.line_thickness // 2)
            xmax = xmin + self.line_thickness
            line_bb = [xmin, col_bb[1], xmax, col_bb[3]]
            line_image = im[col_bb[1]:col_bb[3], xmin:xmax]
            if is_image_black(line_image, min_black_percent=0.1):
                black_cols += 1
            # else:
            #     pdb.set_trace()
            total_cols += 1
        return (total_rows == 0 or black_rows / total_rows > 0.7) and (total_cols == 0 or black_cols/total_cols > 0.7)
    

    def process(self, im: Image, rows, cols, spans, texts, augment_type=None):
        if not self.check(im, rows, cols, spans, texts):
            print('Image not valid for augment')
            return im, rows, cols, spans
        
        im = np.array(im)[:, :, ::-1] # bgr im, opencv format
        im = np.ascontiguousarray(im, dtype=im.dtype)
        orig_im = im.copy()

        orig_cells = self.extract_cells(rows, cols, spans)
        orig_cells = self.texts2cells(texts, orig_cells)
        # choose cand to be anchor cells
        # condition: not span cell
        cand_cells = [cell for cell in orig_cells if cell['relation'][0] >= 0 and cell['relation'][2] >= 0 and not is_span_cell(cell)]
        success = False
        max_try, num_try = 10, 0
        while not success and num_try < max_try:
            is_try_valid = True  # valid if all cells in zone to merge is not span cells

            anchor_cell = np.random.choice(cand_cells)
            start_row_idx, start_col_idx = anchor_cell['relation'][0], anchor_cell['relation'][2]
            # cells on the same row and to the right of anchor cells
            row_cells = [cell for cell in orig_cells if cell['relation'][0] == start_row_idx and cell['relation'][2] >= start_col_idx]
            row_cells.sort(key=lambda cell: cell['relation'][2]) # sort left2right
            col_cells = [cell for cell in orig_cells if cell['relation'][2] == start_col_idx and cell['relation'][0] >= start_row_idx]
            col_cells.sort(key=lambda cell: cell['relation'][0])
            if anchor_cell['relation'][0] == 0:  # if row header cell -> not merge vertically
                num_merge_vertical = 0
            else:
                num_merge_vertical = np.random.randint(1, self.max_merge_vertical)

            if anchor_cell['relation'][0] != 0 and anchor_cell['relation'][2] == 0:  # if col header cell -> not merge horizontally
                num_merge_horizontal = 0
            else:
                num_merge_horizontal = np.random.randint(1, self.max_merge_horizontal)

            # check all cells in zone not span cells
            horizontal_cells = []
            cur_cell_col = None
            for cell in row_cells[:num_merge_horizontal+1]:
                if is_span_cell(cell):
                    break
                if cur_cell_col is not None and cell['relation'][2] != cur_cell_col + 1:  # if cell is not on next col -> break
                    break
                horizontal_cells.append(cell)
                cur_cell_col = cell['relation'][2]
            last_horizontal_cell = horizontal_cells[-1]
            last_col_idx = last_horizontal_cell['relation'][2]

            vertical_cells = []
            cur_cell_row = None
            for cell in col_cells[:num_merge_vertical+1]:
                if is_span_cell(cell):
                    break
                if cur_cell_row is not None and cell['relation'][0] != cur_cell_row + 1:
                    break
                vertical_cells.append(cell)
                cur_cell_row = cell['relation'][0]
            last_vertical_cell = vertical_cells[-1]
            last_row_idx = last_vertical_cell['relation'][0]
            suplement_cells = [cell for cell in orig_cells if start_row_idx < cell['relation'][0] <= last_row_idx and start_col_idx < cell['relation'][2] <= last_col_idx]
            vertical_cells = vertical_cells[1:]  # remove the anchor cell that is already included in horizontal cells
            merge_cells = horizontal_cells + vertical_cells + suplement_cells
            if len(merge_cells) == 1:
                is_try_valid = False
            
            # check if any cells has start row/col exceed the new span bound
            for cell in merge_cells:
                rel = cell['relation']
                if is_span_cell(cell):
                    is_try_valid = False
                if rel[0] < start_row_idx or rel[1] > last_row_idx:
                    is_try_valid = False
                elif rel[2] < start_col_idx or rel[3] > last_col_idx:
                    is_try_valid = False
            
            if not is_try_valid:
                num_try += 1
                print(f'Fail {num_try} times')
                continue
            
            # # debug
            # for cell_idx, cell in enumerate(merge_cells):
            #     bb = cell['bbox']
            #     cx, cy = (bb[0] + bb[2])//2, (bb[1]+bb[3])//2
            #     cv2.putText(im, str(cell_idx), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            # cv2.imwrite('test.png', im)
            # pdb.set_trace()
            

            # get span bb and cell
            span_xmin, span_ymin = anchor_cell['bbox'][0], anchor_cell['bbox'][1]
            span_xmax = last_horizontal_cell['bbox'][2]
            span_ymax = last_vertical_cell['bbox'][3]
            span_bb = [span_xmin, span_ymin, span_xmax, span_ymax]
            min_row, max_row, min_col, max_col = 1e9, -1, 1e9, -1
            for cell in merge_cells:
                rel = cell['relation']
                min_row = min(min_row, rel[0])
                max_row = max(max_row, rel[1])
                min_col = min(min_col, rel[2])
                max_col = max(max_col, rel[3])
            span_cell = {'bbox': span_bb, 'relation': [min_row, max_row, min_col, max_col], 'texts': []}
            span_bg_color = get_background_color(im, span_bb)
            span_roi = im[span_bb[1]:span_bb[3], span_bb[0]:span_bb[2]]
            if not is_image_white(span_roi):
                num_try += 1
                print(f'Fail {num_try} times')
                continue

            # remove all below row lines
            for row_idx in range(start_row_idx, last_row_idx):
                row_cells = [cell for cell in merge_cells if cell['relation'][0] == row_idx]
                xmin, xmax = min([cell['bbox'][0] for cell in row_cells]), max([cell['bbox'][2] for cell in row_cells])
                ymin, ymax = min([cell['bbox'][1] for cell in row_cells]), max([cell['bbox'][3] for cell in row_cells])
                line_ymin = max(0, ymax - self.line_thickness//2)
                line_ymax = line_ymin + self.line_thickness
                line_bb = [xmin, line_ymin, xmax, line_ymax]
                nearby_bb = get_nearby_line_bb(line_bb, axis='horizontal', line_thickness=self.line_thickness)
                nearby_region = orig_im[nearby_bb[1]:nearby_bb[3], nearby_bb[0]:nearby_bb[2]]
                im[line_bb[1]:line_bb[3], line_bb[0]:line_bb[2]] = nearby_region
            
            # remove all right col lines
            for col_idx in range(start_col_idx, last_col_idx):
                col_cells = [cell for cell in merge_cells if cell['relation'][2] == col_idx]
                xmin, xmax = min([cell['bbox'][0] for cell in col_cells]), max([cell['bbox'][2] for cell in col_cells])
                ymin, ymax = min([cell['bbox'][1] for cell in col_cells]), max([cell['bbox'][3] for cell in col_cells])
                line_xmin = max(0, xmax - self.line_thickness//2)
                line_xmax = line_xmin + self.line_thickness
                line_bb = [line_xmin, ymin, line_xmax, ymax]
                nearby_bb = get_nearby_line_bb(line_bb, axis='vertical', line_thickness=self.line_thickness)
                nearby_region = im[nearby_bb[1]:nearby_bb[3], nearby_bb[0]:nearby_bb[2]]
                im[line_bb[1]:line_bb[3], line_bb[0]:line_bb[2]] = nearby_region
            

            # choose cell text to keep
            cells_with_text = [cell for cell in merge_cells if len(cell['texts']) > 0]
            if len(cells_with_text) > 0:
                # choose cell and get roi
                choosen_cell = np.random.choice(cells_with_text)
                choosen_bb = choosen_cell['bbox']
                bb2roi = {}
                for text in choosen_cell['texts']:
                    bb = list(text['bbox'])
                    # narrow text bbox to not include borderline
                    bb = self.narrow_cell_bbox(bb, choosen_bb)
                    roi = im[bb[1]:bb[3], bb[0]:bb[2]].copy()
                    bb2roi[tuple(bb)] = roi
                # mask first
                span_cx, span_cy= (span_xmin+span_xmax)//2, (span_ymin+span_ymax)//2
                for cell in cells_with_text:
                    for text_idx, text in enumerate(cell['texts']):
                        text_bb = text['bbox']
                        text_bb = self.narrow_cell_bbox(text_bb, cell['bbox'])
                        text_w, text_h = text_bb[2] - text_bb[0], text_bb[3] - text_bb[1]
                        im = mask_image(im, text_bb, span_bg_color)
                # paste later
                old_cx, old_cy = (choosen_bb[0]+choosen_bb[2])//2, (choosen_bb[1]+choosen_bb[3])//2
                diff_x, diff_y = span_cx - old_cx, span_cy - old_cy
                for bb, roi in bb2roi.items():
                    new_bb = [bb[0]+diff_x, bb[1]+diff_y, bb[2]+diff_x, bb[3]+diff_y]
                    im[new_bb[1]:new_bb[3], new_bb[0]:new_bb[2]] = roi


            # # change text
            # row2texts = {}
            # merge_cells = self.texts2cells(texts, merge_cells)
            # for cell in merge_cells:
            #     rel = cell['relation']
            #     row_idx = rel[0]
            #     if row_idx not in row2texts:
            #         row2texts[row_idx] = cell['texts']
            #     else:
            #         row2texts[row_idx].extend(cell['texts'])
            # row2height = {}
            # for row_idx, row_texts in row2texts.items():
            #     max_h = max([text['bbox'][3]-text['bbox'][1] for text in row_texts])
            #     row2height[row_idx] = max_h

            # total_h = sum(row2height.values())
            # # get span bb
            # span_xmin, span_ymin = anchor_cell['bbox'][0], anchor_cell['bbox'][1]
            # span_xmax = last_horizontal_cell['bbox'][2]
            # span_ymax = last_vertical_cell['bbox'][3]
            # span_bb = [span_xmin, span_ymin, span_xmax, span_ymax]
            # span_cx, span_cy= (span_xmin+span_xmax)//2, (span_ymin+span_ymax)//2
            # num_rows = len(row2texts)
            # sorted_row_indexes = sorted(list(row2texts.keys()))
            # start_ymin = span_cy - total_h//2
            # span_bg_color = get_background_color(im, span_bb)
            # word_dist = 5
            # # mask first
            # bb2roi = {}
            # for row_idx in sorted_row_indexes:
            #     row_texts = row2texts[row_idx]
            #     row_length = sum([text['bbox'][2]-text['bbox'][0] for text in row_texts]) + int(word_dist * len(row_texts)-1)
            #     start_text_xmin = span_cx - row_length//2
            #     for text_idx, text in enumerate(row_texts):
            #         # get roi
            #         text_bb = text['bbox']
            #         text_w, text_h = text_bb[2] - text_bb[0], text_bb[3] - text_bb[1]
            #         text_roi = im[text_bb[1]:text_bb[3], text_bb[0]:text_bb[2]].copy()
            #         bb2roi[tuple(text_bb)] = text_roi
            #         # mask first
            #         im = mask_image(im, text_bb, span_bg_color)
            
            
            # # cv2.imwrite('test.png', im)
            # # pdb.set_trace()

            # # paste later
            # for row_idx in sorted_row_indexes:
            #     row_texts = row2texts[row_idx]
            #     row_length = sum([text['bbox'][2]-text['bbox'][0] for text in row_texts]) + int(word_dist * len(row_texts)-1)
            #     start_text_xmin = span_cx - row_length//2
            #     for text_idx, text in enumerate(row_texts):
            #         # get new pos
            #         text_bb = text['bbox']
            #         text_w, text_h = text_bb[2] - text_bb[0], text_bb[3] - text_bb[1]
            #         new_bb = [start_text_xmin, start_ymin, start_text_xmin + text_w, start_ymin + text_h]
            #         text_roi = bb2roi[tuple(text_bb)]
            #         # paste
            #         im[new_bb[1]:new_bb[3], new_bb[0]:new_bb[2]] = text_roi
            #         # update pos
            #         start_text_xmin += text_w + word_dist
            #     start_ymin += row2height[row_idx] + word_dist # update row pos

            # update annotations
            # logic: check in remaining cells to determine remain_row and remain_col
            # then sort, then take bounding box from current row index to next row index
            # assumption: all rows have same width, and next row is right after current row
            merge_cell_relations = [cell['relation'] for cell in merge_cells]
            remain_row_indexes, remain_col_indexes = [], []
            for cell in orig_cells + [span_cell]:
                rel = cell['relation']
                if rel in merge_cell_relations: # this cell is already merged
                    continue
                if rel[0] not in remain_row_indexes:
                    remain_row_indexes.append(rel[0])
                if rel[2] not in remain_col_indexes:
                    remain_col_indexes.append(rel[2])

            remain_row_indexes.sort()
            new_rows = []
            for i in range(len(remain_row_indexes)):
                row_idx = remain_row_indexes[i]
                if i < len(remain_row_indexes) - 1:
                    next_row_idx = remain_row_indexes[i+1]
                    new_row_xmin, new_row_ymin = rows[row_idx][:2]
                    new_row_xmax, new_row_ymax = rows[next_row_idx][2], rows[next_row_idx][1]
                else:
                    new_row_xmin, new_row_ymin = rows[row_idx][:2]
                    last_orig_row = rows[-1]
                    new_row_xmax, new_row_ymax = last_orig_row[2], last_orig_row[3]
                new_rows.append([new_row_xmin, new_row_ymin, new_row_xmax, new_row_ymax])
            
            remain_col_indexes.sort()
            new_cols = []
            for i in range(len(remain_col_indexes)):
                col_idx = remain_col_indexes[i]
                if i < len(remain_col_indexes) - 1:
                    next_col_idx = remain_col_indexes[i+1]
                    new_col_xmin, new_col_ymin = cols[col_idx][:2]
                    new_col_xmax, new_col_ymax = cols[next_col_idx][0], cols[next_col_idx][3]
                else:
                    new_col_xmin, new_col_ymin = cols[col_idx][:2]
                    last_orig_col = cols[-1]
                    new_col_xmax, new_col_ymax = last_orig_col[2:]
                new_cols.append([new_col_xmin, new_col_ymin, new_col_xmax, new_col_ymax])

            # filter invalid existing spans
            new_spans = []
            for cell in orig_cells:
                if not is_span_cell(cell):
                    continue
                cell_rel = cell['relation']
                if any([cell_rel[0] < row_idx <= cell_rel[1] for row_idx in remain_row_indexes]) or any([cell_rel[2] < col_idx <= cell_rel[3] for col_idx in remain_col_indexes]):
                    new_spans.append(cell['bbox'])
            # whether to add new span
            span_rel = span_cell['relation']
            if any([span_rel[0] < row_idx <= span_rel[1] for row_idx in remain_row_indexes]) or any([span_rel[2] < col_idx <= span_rel[3] for col_idx in remain_col_indexes]):
                new_spans.append(span_bb)
            
            rows, cols, spans = new_rows, new_cols, new_spans
            success = True


        im = Image.fromarray(im[:, :, ::-1]) # convert back to rgb 
        return im, rows, cols, spans
    


if __name__ == '__main__':
    pass
