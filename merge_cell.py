from utils import *
from base import BaseAugmenter
from mask_line import get_background_color, get_nearby_line_bb

# np.random.seed(39)


class MergeCellAugmenter(BaseAugmenter):
    def __init__(self):
        super().__init__()
        self.max_merge_horizontal = 5 # not include anchor cell yet -> max 6 cells
        self.max_merge_vertical = 5 # not include anchor cell yet -> max 6 cells
        self.line_thickness = 5
    

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


    def merge(self, im: Image, rows, cols, spans, texts):
        im = np.array(im)[:, :, ::-1] # bgr im, opencv format
        orig_im = im.copy()

        cells = self.extract_cells(rows, cols, spans)
        # choose cand to be anchor cells
        # condition: not first row, not first col, and not span cell (to avoid merge header)
        cand_cells = [cell for cell in cells if cell['relation'][0] != 0 and cell['relation'][2] != 0 and not is_span_cell(cell)]
        success = False
        max_try, num_try = 10, 0
        while not success and num_try < max_try:
            is_try_valid = True  # valid if all cells in zone to merge is not span cells

            anchor_cell = np.random.choice(cand_cells)
            start_row_idx, start_col_idx = anchor_cell['relation'][0], anchor_cell['relation'][2]
            # cells on the same row and to the right of anchor cells
            row_cells = [cell for cell in cells if cell['relation'][0] == start_row_idx and cell['relation'][2] >= start_col_idx]
            row_cells.sort(key=lambda cell: cell['relation'][2]) # sort left2right
            col_cells = [cell for cell in cells if cell['relation'][2] == start_col_idx and cell['relation'][0] >= start_row_idx]
            col_cells.sort(key=lambda cell: cell['relation'][0])
            num_merge_horizontal = np.random.randint(1, self.max_merge_horizontal)
            num_merge_vertical = np.random.randint(1, self.max_merge_vertical)
            # check all cells in zone not span cells
            horizontal_cells = []
            for cell in row_cells[:num_merge_horizontal+1]:
                if is_span_cell(cell):
                    break
                horizontal_cells.append(cell)
            last_horizontal_cell = horizontal_cells[-1]
            last_col_idx = last_horizontal_cell['relation'][2]

            vertical_cells = []
            for cell in col_cells[:num_merge_vertical+1]:
                if is_span_cell(cell):
                    break
                vertical_cells.append(cell)
            last_vertical_cell = vertical_cells[-1]
            last_row_idx = last_vertical_cell['relation'][0]
            suplement_cells = [cell for cell in cells if start_row_idx < cell['relation'][0] <= last_row_idx and start_col_idx < cell['relation'][2] <= last_col_idx]
            vertical_cells = vertical_cells[1:]  # remove the anchor cell that is already included in horizontal cells
            merge_cells = horizontal_cells + vertical_cells + suplement_cells
            if len(merge_cells) == 1:
                is_try_valid = False
            
            # check if any cells has start row/col exceed the new span bound
            for cell in merge_cells:
                rel = cell['relation']
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

            
            # remove all below row lines
            for row_idx in range(start_row_idx, last_row_idx):
                cells = [cell for cell in merge_cells if cell['relation'][0] == row_idx]
                xmin, xmax = min([cell['bbox'][0] for cell in cells]), max([cell['bbox'][2] for cell in cells])
                ymin, ymax = min([cell['bbox'][1] for cell in cells]), max([cell['bbox'][3] for cell in cells])
                line_ymin = max(0, ymax - self.line_thickness//2)
                line_ymax = line_ymin + self.line_thickness
                line_bb = [xmin, line_ymin, xmax, line_ymax]
                nearby_bb = get_nearby_line_bb(line_bb, axis='horizontal', line_thickness=self.line_thickness)
                nearby_region = orig_im[nearby_bb[1]:nearby_bb[3], nearby_bb[0]:nearby_bb[2]]
                im[line_bb[1]:line_bb[3], line_bb[0]:line_bb[2]] = nearby_region
            
            # remove all right col lines
            for col_idx in range(start_col_idx, last_col_idx):
                cells = [cell for cell in merge_cells if cell['relation'][2] == col_idx]
                xmin, xmax = min([cell['bbox'][0] for cell in cells]), max([cell['bbox'][2] for cell in cells])
                ymin, ymax = min([cell['bbox'][1] for cell in cells]), max([cell['bbox'][3] for cell in cells])
                line_xmin = max(0, xmax - self.line_thickness//2)
                line_xmax = line_xmin + self.line_thickness
                line_bb = [line_xmin, ymin, line_xmax, ymax]
                nearby_bb = get_nearby_line_bb(line_bb, axis='vertical', line_thickness=self.line_thickness)
                nearby_region = im[nearby_bb[1]:nearby_bb[3], nearby_bb[0]:nearby_bb[2]]
                im[line_bb[1]:line_bb[3], line_bb[0]:line_bb[2]] = nearby_region

            # change text
            row2texts = {}
            merge_cells = self.texts2cells(texts, merge_cells)
            for cell in merge_cells:
                rel = cell['relation']
                row_idx = rel[0]
                if row_idx not in row2texts:
                    row2texts[row_idx] = cell['texts']
                else:
                    row2texts[row_idx].extend(cell['texts'])
            row2height = {}
            for row_idx, row_texts in row2texts.items():
                max_h = max([text['bbox'][3]-text['bbox'][1] for text in row_texts])
                row2height[row_idx] = max_h

            total_h = sum(row2height.values())
            # get span bb
            span_xmin, span_ymin = anchor_cell['bbox'][0], anchor_cell['bbox'][1]
            span_xmax = last_horizontal_cell['bbox'][2]
            span_ymax = last_vertical_cell['bbox'][3]
            span_bb = [span_xmin, span_ymin, span_xmax, span_ymax]
            span_cx, span_cy= (span_xmin+span_xmax)//2, (span_ymin+span_ymax)//2
            num_rows = len(row2texts)
            sorted_row_indexes = sorted(list(row2texts.keys()))
            start_ymin = span_cy - total_h//2
            span_bg_color = get_background_color(im, span_bb)
            word_dist = 5
            # mask first
            bb2roi = {}
            for row_idx in sorted_row_indexes:
                row_texts = row2texts[row_idx]
                row_length = sum([text['bbox'][2]-text['bbox'][0] for text in row_texts]) + int(word_dist * len(row_texts)-1)
                start_text_xmin = span_cx - row_length//2
                for text_idx, text in enumerate(row_texts):
                    # get roi
                    text_bb = text['bbox']
                    text_w, text_h = text_bb[2] - text_bb[0], text_bb[3] - text_bb[1]
                    text_roi = im[text_bb[1]:text_bb[3], text_bb[0]:text_bb[2]].copy()
                    bb2roi[tuple(text_bb)] = text_roi
                    # mask first
                    im = mask_image(im, text_bb, span_bg_color)
            
            # cv2.imwrite('test.png', im)
            # pdb.set_trace()

            # paste later
            for row_idx in sorted_row_indexes:
                row_texts = row2texts[row_idx]
                row_length = sum([text['bbox'][2]-text['bbox'][0] for text in row_texts]) + int(word_dist * len(row_texts)-1)
                start_text_xmin = span_cx - row_length//2
                for text_idx, text in enumerate(row_texts):
                    # get new pos
                    text_bb = text['bbox']
                    text_w, text_h = text_bb[2] - text_bb[0], text_bb[3] - text_bb[1]
                    new_bb = [start_text_xmin, start_ymin, start_text_xmin + text_w, start_ymin + text_h]
                    text_roi = bb2roi[tuple(text_bb)]
                    # paste
                    im[new_bb[1]:new_bb[3], new_bb[0]:new_bb[2]] = text_roi
                    # update pos
                    start_text_xmin += text_w + word_dist
                start_ymin += row2height[row_idx] + word_dist # update row pos

            # update annotations
            spans.append(span_bb)
            success = True

        im = Image.fromarray(im[:, :, ::-1]) # convert back to rgb 
        return im, rows, cols, spans


    def check(self, im: Image, rows, cols, spans, texts):
        """
            check if the image is valid for augmentation
            condition:
             + must have border
        """
        return True
    

    def process(self, im: Image, rows, cols, spans, texts):
        if not self.check(im, rows, cols, spans, texts):
            return im, rows, cols, spans
        
        im, rows, cols, spans = self.merge(im, rows, cols, spans, texts)
        return im, rows, cols, spans
    


if __name__ == '__main__':
    pass
