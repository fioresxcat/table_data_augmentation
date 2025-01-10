from utils import *

def nothing():
    src_dir = '/home/fiores/Desktop/VNG/table_recognition/pdf_table_imgs_tungtx/pdf_table_imgs_tungtx'
    dir = 'temp_pdf_imgs'
    src_xpaths = sorted([fp for fp in Path(src_dir).rglob('*.xml')])
    src_xnames = [xp.name for xp in src_xpaths]
    for ip in Path(dir).glob('*'):
        if not is_image(ip):
            continue
        xname = f'{ip.stem}.xml'
        if xname in src_xnames:
            index = src_xnames.index(xname)
            xp = src_xpaths[index]
            shutil.copy(xp, dir)
            print(f'copied {xp}')
            



if __name__ == '__main__':
    pass
    nothing()
    