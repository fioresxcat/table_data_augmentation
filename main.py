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

    dir = 'temp_imgs_2'
    out_dir = 'results/temp_imgs_2'
    os.makedirs(out_dir, exist_ok=True)
    for ip in Path(dir).glob('*'):
        # if '150' not in ip.stem:
        #     continue
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
        rows.sort(key=lambda x: x[1])
        cols.sort(key=lambda x: x[0])
        spans.sort(key=lambda x: x[1])

        for i in range(1):
            print(f'Augment file {ip}, time {i}')
            im, rows, cols, spans = augmenter.process(im, rows, cols, spans, texts, augment_type='all_rows')

            # im.save(os.path.join(out_dir, f'{ip.stem}.png'))
            # boxes = rows + cols + spans
            # names = ['row']*len(rows) + ['col']*len(cols) + ['span']*len(spans)
            # write_to_xml(boxes, names, im.size, os.path.join(out_dir, f'{ip.stem}.xml'))
            # pdb.set_trace()
        
        im.save('test.png')
        pdb.set_trace()

        # draw
        # draw = ImageDraw.Draw(im)
        # for row in rows:
        #     draw.rectangle(row, outline='red', width=2)
        # for col in cols:
        #     draw.rectangle(col, outline='green', width=2)
        # for span in spans:
        #     draw.rectangle(span, outline='blue', width=2)

        # im.save(os.path.join(out_dir, f'{ip.stem}.png'))
        # boxes = rows + cols + spans
        # names = ['row']*len(rows) + ['col']*len(cols) + ['span']*len(spans)
        # write_to_xml(boxes, names, im.size, os.path.join(out_dir, f'{ip.stem}.xml'))



if __name__ == '__main__':
    pass
    main()
