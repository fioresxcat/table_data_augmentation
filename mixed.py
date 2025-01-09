from mask_line import MaskLineAugmenter
from change_color import ChangeColorAugmenter
from merge_cell import MergeCellAugmenter
from base import BaseAugmenter
from utils import *

class MixedAugmenter(BaseAugmenter):
    def __init__(self):
        self.mask_line = MaskLineAugmenter()
        self.change_color = ChangeColorAugmenter()
        self.merge_cell = MergeCellAugmenter()

    
    def process(self, im: Image, rows, cols, spans, texts):
        aug_type = np.random.choice(['type1', 'type2'])
        aug_type = 'type2'

        if aug_type == 'type1':
            im, rows, cols, spans = self.mask_line.process(im, rows, cols, spans, texts)
            im, rows, cols, spans = self.change_color.process(im, rows, cols, spans, texts)
        elif aug_type == 'type2':
            im, rows, cols, spans = self.merge_cell.process(im, rows, cols, spans, texts)
            im, rows, cols, spans = self.change_color.process(im, rows, cols, spans, texts)

        return im, rows, cols, spans
    


if __name__ == '__main__':
    pass

