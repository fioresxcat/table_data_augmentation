from utils import *

def nothing():
    im = Image.open('/home/fiores/Pictures/Screenshot from 2025-01-06 17-20-31.png')
    im = im.convert('L').convert('RGB')
    im.save('test.png')


if __name__ == '__main__':
    pass
    nothing()
    