from base import BaseAugmenter
from utils import *
from scipy.stats import mode
from change_color import find_streaks

# np.random.seed(42)

def get_nearby_line_bb(line_bb, line_thickness=5, axis='vertical', side='left'):
    if axis == 'vertical':
        if line_bb[0] >= line_thickness:
            xmin = line_bb[0] - line_thickness
            xmax = line_bb[0]
        else:
            # xmin = bbox[2]
            # xmax = bbox[2] + self.line_thickness
            xmin, xmax = line_bb[0], line_bb[2]
        ymin = line_bb[1]
        ymax = line_bb[3]
    elif axis == 'horizontal':
        if line_bb[1] >= line_thickness:
            ymin = line_bb[1] - line_thickness
            ymax = line_bb[1]
        else:
            # ymin = bbox[3]
            # ymax = bbox[3] + self.line_thickness
            ymin, ymax = line_bb[1], line_bb[3]
        xmin = line_bb[0]
        xmax = line_bb[2]
    return xmin, ymin, xmax, ymax



def get_background_color(image, bbox):
    """
        Get the background color of a table cell in a grayscale image.
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    x, y, w, h = bbox
    # Crop the region of interest (ROI)
    roi = image[y:y+h, x:x+w]
    
    # Apply Otsu's thresholding to segment text and background
    _, binary_mask = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Extract background pixels (where binary_mask == 0)
    background_pixels = roi[binary_mask == 0]
    
    # Calculate the mode (most common intensity) of background pixels
    if len(background_pixels) > 0:
        background_color = mode(background_pixels, axis=None, keepdims=True).mode[0]
    else:
        # Fallback: Use the mean if no background pixels are detected
        background_color = int(np.mean(roi))
    
    return background_color


class MaskLineAugmenter(BaseAugmenter):
    def __init__(self):
        super().__init__()
        self.line_thickness = 4
        self.augment_types = ['all_rows', 'all_cols', 'all_rows_cols', 'random_rows']


    def check(self, im: Image, rows, cols, spans, texts):
        """
            check if the image is valid for this augmentation
            condition:
             + must have black border line
        """
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
            total_cols += 1
        return (total_rows == 0 or black_rows / total_rows > 0.7) and (total_cols == 0 or black_cols/total_cols > 0.7)
    

    def mask_rows(self, im, rows, cols, spans, text_boxes, cells, remove_indexes):
        """
            remove upper row edges
        """
        rows = sorted(rows, key=lambda x: x[1])
        for row_idx, row_bb in enumerate(rows):
            if row_idx not in remove_indexes:
                continue
            row_bb = list(map(int, row_bb))

            overlap_span_cells = []
            for cell in cells:
                if not is_span_cell(cell):
                    continue
                span_bb = cell['bbox']
                if abs(row_bb[1]-span_bb[1] > 5):
                    r1, r2, iou = iou_axis(row_bb[1], row_bb[3], span_bb[1], span_bb[3])
                    if r1 > 0.7:
                        overlap_span_cells.append(cell)


            # # ---- split by span cells ----
            # if len(overlap_spans) > 0:
            #     overlap_spans.sort(key=lambda x: x[0])
            #     valid_ranges = [(row_bb[0], overlap_spans[0][0])]
            #     for span_idx in range(len(overlap_spans)-1):
            #         cur_span = overlap_spans[span_idx]
            #         next_span = overlap_spans[span_idx+1]
            #         if next_span[0] > cur_span[2]:
            #             valid_ranges.append((cur_span[2], next_span[0]))
            #     valid_ranges.append((overlap_spans[-1][2], row_bb[2]))
            # else:
            #     valid_ranges = [(row_bb[0], row_bb[2])]
            # valid_ranges = [list(map(int, el)) for el in valid_ranges if el[1] > el[0]]
            # for temp_idx, valid_range in enumerate(valid_ranges):
            #     valid_range[0] += self.line_thickness//2
            #     valid_range[1] -= self.line_thickness//2
            #     valid_ranges[temp_idx] = valid_range

            # ---- split by columns ----
            span_cols = []
            for cell in overlap_span_cells:
                for col_idx in range(cell['relation'][2], cell['relation'][3]+1):
                    span_cols.append(col_idx)
            cols = sorted(cols, key=lambda x: x[0])
            # valid_ranges = [(0, cols[0][1])]
            valid_ranges = []
            for col_idx in range(len(cols)-1):
                if col_idx in span_cols:
                    continue
                cur_col = cols[col_idx]
                next_col = cols[col_idx+1]
                valid_ranges.append((cur_col[0], next_col[0]))
            if len(cols) - 1 not in span_cols:
                valid_ranges.append((cols[-1][0], row_bb[2]))
            valid_ranges = [list(map(int, el)) for el in valid_ranges if el[1] > el[0]]
            for temp_idx, valid_range in enumerate(valid_ranges):
                valid_range[0] = min(valid_range[0]+1, valid_range[1]-1)
                valid_range[1] = max(valid_range[1]-1, valid_range[0]+1)
                valid_ranges[temp_idx] = valid_range

            # ---- remove upper edge ----
            for x_range in valid_ranges:
                xmin, xmax = x_range
                ymin = max(0, row_bb[1] - self.line_thickness // 2)
                ymax = ymin + self.line_thickness
                line_bb = [xmin, ymin, xmax, ymax]
                nearby_bb = get_nearby_line_bb(line_bb=line_bb, line_thickness=self.line_thickness, axis='horizontal')
                # check overlap with any text bbox
                is_overlap = False
                if row_idx > 0:
                    for text_bb in text_boxes:
                        text_xmin, text_ymin, text_xmax, text_ymax = text_bb
                        xr1, xr2, xiou = iou_axis(text_xmin, text_xmax, nearby_bb[0], nearby_bb[2])
                        if xr1 > 0:
                            yr1, yr2, yiou = iou_axis(text_ymin, text_ymax, nearby_bb[1], nearby_bb[3])
                            if yr2 > 0.5:
                                is_overlap = True
                                # print(f'overlap row {row_idx}, yr2: {yr2}')
                                break
                if not is_overlap:
                    nearby_region = im[nearby_bb[1]:nearby_bb[3], nearby_bb[0]:nearby_bb[2]]
                else:
                    temp_bb = [xmin, row_bb[1], xmax, row_bb[3]]
                    bg_color = get_background_color(im, temp_bb)
                    h, w = nearby_bb[3] - nearby_bb[1], nearby_bb[2] - nearby_bb[0]
                    nearby_region = np.full(shape=(h, w, 3), fill_value=bg_color, dtype=np.uint8)
                im[ymin:ymax, xmin:xmax] = nearby_region
        return im
    

    def mask_cols(self, im, rows, cols, spans, text_boxes, cells, mask_type: Literal['all', 'random']='all'):
        """
            remove left column edges
        """
        cols = sorted(cols, key=lambda x: x[0])
        if mask_type == 'random':
            remove_indexes = np.random.choice(list(range(len(cols))), size=int(len(cols) * 0.5), replace=False)
        else:
            remove_indexes = list(range(len(cols)))
        
        for col_idx, col_bb in enumerate(cols):
            if col_idx not in remove_indexes:
                continue
            col_bb = list(map(int, col_bb))

            overlap_span_cells = []
            for cell in cells:
                if not is_span_cell(cell):
                    continue
                span_bb = cell['bbox']
                if abs(span_bb[0]-col_bb[0]) > 5:
                    c1, c2, iou = iou_axis(col_bb[0], col_bb[2], span_bb[0], span_bb[2])
                    if c1 > 0.7:
                        overlap_span_cells.append(cell)

            # # ---- split by span cells ----
            # if len(overlap_spans) > 0:
            #     overlap_spans.sort(key=lambda x: x[1])
            #     valid_ranges = [(col_bb[1], overlap_spans[0][1])]
            #     for span_idx in range(len(overlap_spans)-1):
            #         cur_span = overlap_spans[span_idx]
            #         next_span = overlap_spans[span_idx+1]
            #         if next_span[1] > cur_span[3]:
            #             valid_ranges.append((cur_span[3], next_span[1]))
            #     valid_ranges.append((overlap_spans[-1][3], col_bb[3]))
            # else:
            #     valid_ranges = [(col_bb[1], col_bb[3])]
            # valid_ranges = [list(map(int, el)) for el in valid_ranges if el[1] > el[0]]

            # ---- split by rows and spans ----
            span_rows = []
            for cell in overlap_span_cells:
                for row_idx in range(cell['relation'][0], cell['relation'][1]+1):
                    span_rows.append(row_idx)
            rows = sorted(rows, key=lambda x: x[1])
            valid_ranges = []
            for row_idx in range(len(rows)-1):
                if row_idx in span_rows:
                    continue
                cur_row = rows[row_idx]
                next_row = rows[row_idx+1]
                valid_ranges.append((cur_row[1], next_row[1]))
            if len(rows) - 1 not in span_rows:
                valid_ranges.append((rows[-1][1], col_bb[3]))
            valid_ranges = [list(map(int, el)) for el in valid_ranges if el[1] > el[0]]
            for temp_idx, valid_range in enumerate(valid_ranges):
                valid_range[0] = min(valid_range[0]+1, valid_range[1]-1)
                valid_range[1] = max(valid_range[1]-1, valid_range[0]+1)
                valid_ranges[temp_idx] = valid_range

            # ---- remove left edge ----
            for y_range in valid_ranges:
                xmin = max(0, col_bb[0] - self.line_thickness // 2)
                xmax = xmin + self.line_thickness
                ymin, ymax = y_range
                line_bb = [xmin, ymin, xmax, ymax]
                nearby_bb = get_nearby_line_bb(line_bb=line_bb, line_thickness=self.line_thickness, axis='vertical')
                # check overlap with any text bbox
                is_overlap = False
                if col_idx > 0:
                    for text_bb in text_boxes:
                        text_xmin, text_ymin, text_xmax, text_ymax = text_bb
                        yr1, yr2, yiou = iou_axis(text_ymin, text_ymax, nearby_bb[1], nearby_bb[3])
                        if yr1 > 0:
                            xr1, xr2, xiou = iou_axis(text_xmin, text_xmax, nearby_bb[0], nearby_bb[2])
                            if xr2 > 0.5:
                                is_overlap = True
                                break
                if not is_overlap:
                    nearby_region = im[nearby_bb[1]:nearby_bb[3], nearby_bb[0]:nearby_bb[2]]
                else:
                    temp_bb = [col_bb[0], ymin, col_bb[2], ymax]
                    bg_color = get_background_color(im, temp_bb)
                    # bg_color = (255, 255, 255)
                    h, w = nearby_bb[3] - nearby_bb[1], nearby_bb[2] - nearby_bb[0]
                    nearby_region = np.full(shape=(h, w, 3), fill_value=bg_color, dtype=np.uint8)
                im[ymin:ymax, xmin:xmax] = nearby_region

        return im
    


    def mask_random_rows(self, im: Image, rows, cols, spans, text_boxes, cells):
        """
            only keep some row border lines
            idea:
            + find streaks
            + augment in each streak
            + divide streak based on length
            + augment in each streak zone
            condition:
             + only apply to row streaks with no span cell inside

            divide_indexes: [1, a, b, n]
            -> zones: (1 -> a), (a+1 -> b), (b+1 -> n) (inclusive at both end)
        """

        def is_divide_indexes_valid(divide_indexes):
            for i, index in enumerate(divide_indexes):
                if i == len(divide_indexes) - 1:
                    break
                next_index = divide_indexes[i+1]
                if i == 0 and next_index - index < 1:
                    return False
                elif next_index - (index+1) < 1:
                    return False
            return True

        if len(rows) <= 1:
            return im
        
        valid_row_indexes = [row_idx for row_idx, row in enumerate(rows) if is_row_valid(row, spans)]
        cells = self.extract_cells(rows, cols, spans)
        streaks = find_streaks(valid_row_indexes, min_len=2)
        
        for row_indexes in streaks:
            num_rows = len(row_indexes)
            if num_rows <= 3:
                num_divide = 1
            elif 3 < num_rows <= 6:
                num_divide = 2
            elif 6 < num_rows <= 15:
                num_divide = np.random.choice([2,3], p=[0.5,0.5])
                # num_divide = 3
            else:
                num_divide = 4
            
            if num_divide == 1:
                divide_indexes = [row_indexes[0], row_indexes[-1]]
            else:
                divide_indexes = list(np.random.choice(row_indexes, size=num_divide-1))
                divide_indexes.sort()
                divide_indexes = [row_indexes[0]] + divide_indexes + [row_indexes[-1]]
                num_try, max_try = 0, 10
                while not is_divide_indexes_valid(divide_indexes) and num_try <= max_try:
                    divide_indexes = list(np.random.choice(row_indexes, size=num_divide-1))
                    divide_indexes.sort()
                    divide_indexes = [row_indexes[0]] + divide_indexes + [row_indexes[-1]]
                    num_try += 1
                if num_try > max_try:
                    return im
                
            for i in range(len(divide_indexes)-1):
                if i == 0:
                    start_row_index = divide_indexes[i]
                    end_row_index = divide_indexes[i+1]
                else:
                    start_row_index = divide_indexes[i] + 1
                    end_row_index = divide_indexes[i+1]
                start_row_bb = rows[start_row_index]
                end_row_bb = rows[end_row_index]
                # mask all inside row borders (note: do not mask the first row - header)
                im = self.mask_rows(im, rows, cols, spans, text_boxes, cells, remove_indexes=list(range(max(2, start_row_index+1), end_row_index+1)))  # mask upper line

                # mask cols (left line)
                for col_idx, col_bb in enumerate(cols):
                    xmin = max(0, col_bb[0] - self.line_thickness // 2)
                    xmax = xmin + self.line_thickness
                    ymin, ymax = start_row_bb[1]+1, end_row_bb[3]-1
                    line_bb = [xmin, ymin, xmax, ymax]
                    nearby_bb = get_nearby_line_bb(line_bb=line_bb, line_thickness=self.line_thickness, axis='vertical')
                    # check overlap with any text bbox
                    is_overlap = False
                    if col_idx > 0:
                        for text_bb in text_boxes:
                            text_xmin, text_ymin, text_xmax, text_ymax = text_bb
                            yr1, yr2, yiou = iou_axis(text_ymin, text_ymax, nearby_bb[1], nearby_bb[3])
                            if yr1 > 0:
                                xr1, xr2, xiou = iou_axis(text_xmin, text_xmax, nearby_bb[0], nearby_bb[2])
                                if xr2 > 0.5:
                                    is_overlap = True
                                    break
                    if not is_overlap:
                        nearby_region = im[nearby_bb[1]:nearby_bb[3], nearby_bb[0]:nearby_bb[2]]
                    else:
                        temp_bb = [col_bb[0], ymin, col_bb[2], ymax]
                        bg_color = get_background_color(im, temp_bb)
                        # bg_color = (255, 255, 255)
                        h, w = nearby_bb[3] - nearby_bb[1], nearby_bb[2] - nearby_bb[0]
                        nearby_region = np.full(shape=(h, w, 3), fill_value=bg_color, dtype=np.uint8)
                    im[ymin:ymax, xmin:xmax] = nearby_region

        return im
    


    def process(self, im: Image, rows, cols, spans, texts, augment_type='all_rows_cols'):
        assert augment_type in self.augment_types, f'{augment_type} not supported!'

        im = np.array(im)[:, :, ::-1] # convert to bgr, cv2 format
        im = np.ascontiguousarray(im)
        text_boxes = [text['bbox'] for text in texts]
        cells = self.extract_cells(rows, cols, spans)

        if augment_type == 'all_rows':
            remove_indexes = list(range(len(rows)))
            im = self.mask_rows(im, rows, cols, spans, text_boxes, cells, remove_indexes)
        elif augment_type == 'all_cols':
            remove_indexes = list(range(len(cols)))
            im = self.mask_cols(im, rows, cols, spans, text_boxes, cells, remove_indexes)
        elif augment_type == 'all_rows_cols':
            im = self.mask_rows(im, rows, cols, spans, text_boxes, cells, remove_indexes=list(range(len(rows))))
            im = self.mask_cols(im, rows, cols, spans, text_boxes, cells, remove_indexes=list(range(len(cols))))
        elif augment_type == 'random_rows':
            im = self.mask_random_rows(im, rows, cols, spans, text_boxes, cells)
        # elif augment_type == 'random_cols':
        #     im = self.mask_cols(im, rows, cols, spans, text_boxes, mask_type='random')
        return Image.fromarray(im[:, :, ::-1]), rows, cols, spans
    


if __name__ == "__main__":
    pass

    augmenter = MaskLineAugmenter()