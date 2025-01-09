from mask_line import MaskLineAugmenter
from change_color import ChangeColorAugmenter
from merge_cell import MergeCellAugmenter
from mixed import MixedAugmenter
from utils import *


def main():
    # augmenter = MixedAugmenter()
    augmenter = MaskLineAugmenter()
    # augmenter = ChangeColorAugmenter()
    # augmenter = MergeCellAugmenter()
    dir = 'temp_pdf_imgs'
    for ip in Path(dir).glob('*'):
        if '0._bao_cao_chinh_158_table_0' not in ip.stem:
            continue
        if not is_image(ip):
            continue
        xp = ip.with_suffix('.xml')
        if not xp.exists():
            continue
        jp = ip.with_suffix('.json')
        if jp.exists():
            with open(jp) as f:
                data = json.load(f)
            texts = []
            for shape in data['shapes']:
                text = {
                    'bbox': poly2box(shape['points']),
                    'text': shape['text'] if 'text' in shape else '' 
                }
                texts.append(text)
        else:
            texts = []
            
        im = Image.open(ip)
        boxes, names = parse_xml(xp)
        rows, cols, spans = get_bb_type(boxes, names)
        im_aug, rows, cols, spans = augmenter.process(im, rows, cols, spans, texts, augment_type='random_cols')
        im_aug.save('test.png')



if __name__ == '__main__':
    pass
    main()
