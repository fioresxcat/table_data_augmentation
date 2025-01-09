from utils import *

class BaseAugmenter:
    def __init__(self):
        pass
    

    def extract_cells(self, rows, cols, spans):
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


    def is_poly_belong(self, text, block):
        xmin_block, ymin_block, xmax_block, ymax_block = block
        xmin_text, ymin_text, xmax_text, ymax_text = text
        x_center = xmin_text + (xmax_text - xmin_text)/2.0
        y_center = ymin_text + (ymax_text - ymin_text)/2.0
        if x_center > xmin_block and x_center < xmax_block and y_center > ymin_block and y_center < ymax_block:
            return True
        return False
    

    def texts2cells(self, texts, cells):
        '''
        texts: a list, format of each element is {'box': ..., 'score':..., 'roi':..., 'text':...}
        cells: a list, format of each element is {'box': ..., 'relative': ...}
        '''
        mask = [0 for i in range(len(texts))]
        for cell in cells:
            cell_texts = [] 
            for i, text in enumerate(texts):
                if mask[i] == 1: continue
                if self.is_poly_belong(text['bbox'], cell['bbox']):
                    cell_texts.append(text)
                    mask[i] = 1

            # sort text
            if len(cell_texts) > 0:
                bbs = []
                bb2text = {}
                for text in cell_texts:
                    bb = text['bbox']
                    bbs.append(bb)
                    bb2text[tuple(bb)] = text
                sorted_bbs, _ = sort_bbs(bbs)
                sorted_texts = [bb2text[tuple(bb)] for bb in sorted_bbs]
                cell['texts'] = sorted_texts
            else:
                cell['texts'] = []
        return cells