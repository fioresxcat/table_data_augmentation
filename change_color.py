from utils import *
from base import BaseAugmenter

# np.random.seed(42)


def longest_consecutive_streak(list_numbers):
    if not list_numbers:
        return []
    
    # Sort the list to ensure consecutive numbers are adjacent
    list_numbers = sorted(list_numbers)
    
    # Initialize variables to track the longest streak
    longest_streak = []
    current_streak = [list_numbers[0]]
    
    for i in range(1, len(list_numbers)):
        if list_numbers[i] == list_numbers[i - 1] + 1:
            # If consecutive, add to the current streak
            current_streak.append(list_numbers[i])
        else:
            # If not consecutive, update longest streak if needed
            if len(current_streak) > len(longest_streak):
                longest_streak = current_streak
            # Reset current streak
            current_streak = [list_numbers[i]]
    
    # Final check for the last streak
    if len(current_streak) > len(longest_streak):
        longest_streak = current_streak
    
    return longest_streak


def find_streaks(nums, min_len=2):
    if not nums:
        return []

    nums.sort()  # Sort the numbers to ensure consecutive numbers are adjacent
    streaks = []
    current_streak = [nums[0]]  # Initialize the first streak

    for i in range(1, len(nums)):
        if nums[i] == nums[i - 1] + 1:  # Check if the current number is consecutive
            current_streak.append(nums[i])
        else:
            if len(current_streak) >= min_len:  # If the streak is valid, add it to the result
                streaks.append(current_streak)
            current_streak = [nums[i]]  # Start a new streak

    # Check the last streak
    if len(current_streak) >= min_len:
        streaks.append(current_streak)

    return streaks


class ChangeColorAugmenter(BaseAugmenter):
    def __init__(self):
        super().__init__()
        self.color_dict = {
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'gray': (128, 128, 128),
            'white': (255, 255, 255)
        }
        self.alpha_factor = 0.6
        self.transparent_perc = 0.5
        self.augmenter = {
            'sole_rows': self.change_color_sole_rows,
            'sole_cols': self.change_color_sole_cols,
            'header': self.change_color_header,
            'sub_header': self.change_color_sub_header,
            'sole_cells': self.change_color_sole_cells
        }
        self.offset = 10
        self.header_color = (15, 15, 15)  # a bit darker
        self.subheader_color = (100, 100, 100) # lighter a bit
        self.gray_color = (128, 128, 128) # gray
    

    def get_color(self, name=None):
        if name is None:
            name = np.random.choice(list(self.color_dict.keys()))
        return self.color_dict[name]
    


    def random_shift_color(self, color):
        assert len(color) == 3, f'Color must be rgb color with 3 values'
        return (color[0]+np.random.randint(-self.offset, self.offset), color[1]+np.random.randint(-self.offset, self.offset), color[2]+np.random.randint(-self.offset, self.offset))


    def get_overlay_im(self, size, color):
        im = Image.new('RGBA', size, color + (int(self.transparent_perc*255),))
        return im
    

    def paste(self, bg, roi, pos, alpha_factor=None, mask=None):
        orig_bg_mode = bg.mode
        roi = roi.convert('RGBA')
        bg = bg.convert('RGBA')
        if alpha_factor is None:
            alpha_factor = self.alpha_factor
        alpha = roi.split()[-1].point(lambda x: x * alpha_factor)
        if mask is not None:
            alpha = Image.fromarray(mask * np.array(alpha))
        roi.putalpha(alpha)
        bg.alpha_composite(roi, pos)
        return bg.convert(orig_bg_mode)
    

    def change_color_sole_rows(self, im, rows, cols, spans, cells, valid_row_indexes):
        indexes = longest_consecutive_streak(valid_row_indexes)
        if len(indexes) >= 3 or len(indexes) / len(rows) >= 0.8:
            color1, color2 = self.get_color('white'), self.get_color('gray')
            colors = [color1 if i%2 == 0 else color2 for i in range(len(indexes))]
            assert len(colors) == len(indexes)
            for row_idx, color in zip(indexes, colors):
                bb = rows[row_idx]
                bb_w, bb_h = bb[2] - bb[0], bb[3] - bb[1]
                overlay_im = Image.new('RGBA', (bb_w, bb_h), color + (128,))
                im = self.paste(im, overlay_im, (bb[0], bb[1]), alpha_factor=0.65)
        return im
    

    def change_color_sole_cols(self, im, rows, cols, spans, cells, valid_row_indexes):
        pass


    def change_color_header(self, im, rows, cols, spans, cells, valid_row_indexes):
        first_row = rows[0]
        if is_row_valid(first_row, spans):
            color = self.get_color('gray')
            bb_w, bb_h = first_row[2] - first_row[0], first_row[3] - first_row[1]
            overlay_im = self.get_overlay_im((bb_w, bb_h), color)
            im = self.paste(im, overlay_im, (first_row[0], first_row[1]), alpha_factor=self.alpha_factor)

        return im
    

    def change_color_sub_header(self, im, rows, cols, spans, cells, valid_row_indexes):
        for row in rows:
            for span in spans:
                if is_box_is_span(row, span):
                    bb_w, bb_h = row[2] - row[0], row[3] - row[1]
                    color = self.get_color('gray')
                    overlay_im= self.get_overlay_im((bb_w, bb_h), color)
                    im = self.paste(im, overlay_im, (row[0], row[1]), alpha_factor=self.alpha_factor)
        
        return im


    def change_color_sole_cells(self, im, rows, cols, spans, cells, valid_row_indexes):
        """
            change color for cells
            condition: cells on normal rows, condition same as so le row cho chắc
        """
        indexes = longest_consecutive_streak(valid_row_indexes)
        valid_rows = [rows[idx] for idx in indexes]
        if len(indexes) >= 3 or len(indexes) / len(rows) >= 0.8:
            for row_idx in indexes:
                row_cells = [cell for cell in cells if cell['relation'][0] == row_idx]
                colors = ['white' if i%2==0 else 'gray' for i in range(len(row_cells))]
                for cell, color_name in zip(row_cells, colors):
                    color = self.get_color(color_name)
                    bb_w, bb_h = get_bb_size(cell['bbox'])
                    overlay_im = self.get_overlay_im((bb_w, bb_h), color)
                    im = self.paste(im, overlay_im, tuple(cell['bbox'][:2]))
        return im



    # def process(self, im: Image, rows, cols, spans, augment_type=None):
    #     """
    #         2 type
    #         + so le dòng / cột
    #          chọn ra 2 màu -> overllay lên 2 dòng so le nhau
    #         + header màu khác
    #         + so le cell
    #     """
    #     rows = sorted(rows, key=lambda x: x[1])
    #     cols= sorted(cols, key=lambda x: x[0])

    #     # find all row that not overlap with any span cells
    #     valid_row_indexes = [row_idx for row_idx, row in enumerate(rows) if is_row_valid(row, spans)]
    #     cells = self.extract_cells(rows, cols, spans)

    #     if augment_type is None:
    #         augment_type = np.random.choice(list(self.augmenter.keys()))
    #     assert augment_type in list(self.augmenter.keys())
    #     augment_func = self.augmenter[augment_type]
    #     im = augment_func(im, rows, cols, spans, cells, valid_row_indexes)

    #     return im, rows, cols, spans

    def check(self, im: Image, rows, cols, spans, texts):
        """
            check if the image is valid for augmentation
            condition:
        """
        return True
    
    
    def process(self, im: Image, rows, cols, spans, texts, augment_type=None):
        """
            + so le dòng / cột
             chọn ra 2 màu -> overllay lên 2 dòng so le nhau
            + header màu khác
            + so le cell
        """
        if not self.check(im, rows, cols, spans, texts):
            return im, rows, cols, spans

        im = im.convert('L').convert('RGB') # convert to gray image first
        rows = sorted(rows, key=lambda x: x[1])
        cols = sorted(cols, key=lambda x: x[0])

        # find all row that not overlap with any span cells
        valid_row_indexes = [row_idx for row_idx, row in enumerate(rows) if is_row_valid(row, spans)]
        cells = self.extract_cells(rows, cols, spans)

        # ------------- change header color --------------
        if np.random.rand() < 0.9:
            first_row = rows[0]
            if is_row_valid(first_row, spans):
                bb_w, bb_h = first_row[2] - first_row[0], first_row[3] - first_row[1]
                overlay_im = self.get_overlay_im((bb_w, bb_h), self.random_shift_color(self.header_color))
                im = self.paste(im, overlay_im, (first_row[0], first_row[1]), alpha_factor=self.alpha_factor)
        
        # ------------- change subheader color ------------
        # condition: row == span will be considered as subheader
        if np.random.rand() < 0.9:
            for row in rows:
                for span in spans:
                    if not is_box_is_span(row, span): 
                        continue
                    bb_w, bb_h = row[2] - row[0], row[3] - row[1]
                    overlay_im= self.get_overlay_im((bb_w, bb_h), self.random_shift_color(self.subheader_color))
                    im = self.paste(im, overlay_im, (row[0], row[1]), alpha_factor=self.alpha_factor)
        
        # ------------- change sole row color -------------
        if np.random.rand() < 0.95:
            if np.random.rand() < 0.5:
                streaks = find_streaks(valid_row_indexes, min_len=2)
                color1, color2 = self.get_color('gray'), self.get_color('white')
                color1, color2 = self.random_shift_color(color1), self.random_shift_color(color2)
                for row_indexes in streaks:
                    colors = [color1 if i%2 == 0 else color2 for i in range(len(row_indexes))]
                    assert len(colors) == len(row_indexes)
                    for row_idx, color in zip(row_indexes, colors):
                        if row_idx == 0:
                            continue
                        bb = rows[row_idx]
                        bb_w, bb_h = bb[2] - bb[0], bb[3] - bb[1]
                        overlay_im = Image.new('RGBA', (bb_w, bb_h), color + (128,))
                        im = self.paste(im, overlay_im, (bb[0], bb[1]), alpha_factor=self.alpha_factor)
            

            # hoặc ------------- change sole cell color ---------------
            # note: exclude first row
            else:
                streaks = find_streaks(valid_row_indexes, min_len=2)
                color1, color2 = self.get_color('white'), self.get_color('gray')
                color1, color2 = self.random_shift_color(color1), self.random_shift_color(color2)
                for row_indexes in streaks:
                    for row_idx in row_indexes:
                        if row_idx == 0:
                            continue
                        row_cells = [cell for cell in cells if cell['relation'][0] == row_idx and cell['relation'][2] >= 0]
                        colors = [color1 if i%2==0 else color2 for i in range(len(row_cells))]
                        for cell, color in zip(row_cells, colors):
                            bb_w, bb_h = get_bb_size(cell['bbox'])
                            overlay_im = self.get_overlay_im((bb_w, bb_h), color)
                            im = self.paste(im, overlay_im, tuple(cell['bbox'][:2]), alpha_factor=self.alpha_factor)
        
        return im, rows, cols, spans


if __name__ == '__main__':
    pass