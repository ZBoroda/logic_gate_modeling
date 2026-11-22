"""Cut the panels that are not data plots (network schematics, genotype
neighbourhoods, the fixed-N histograms of Fig. S2) out of the paper PDF, unchanged.

  python -m paper_extended.crop_panels path/to/2610.40635.pdf   ->  paper_extended/panels/
"""
import os
import sys

import fitz  # PyMuPDF

HERE = os.path.dirname(__file__)
# name: (page, x0, y0, x1, y1) in PDF points (1/72 inch)
CROPS = {
    'fig3A': (7, 196, 157, 331, 262),
    'fig4A': (10, 150, 62, 302, 496),
    'fig4D': (10, 313, 340, 545, 501),
    'fig5A': (12, 140, 128, 222, 348),
    'fig5D': (12, 177, 294, 388, 440),
    'figS2A': (20, 141, 407, 289, 560),
    'figS2B': (20, 292, 407, 437, 560),
}
# parts of a neighbouring panel inside a crop, blanked: name -> (x0, y0, x1, y1) in points
BLANK = {'fig5D': (177, 294, 195, 360)}   # the O box of Fig. 5A


def main(pdf):
    d = fitz.open(pdf)
    os.makedirs(os.path.join(HERE, 'panels'), exist_ok=True)
    for name, (page, *box) in CROPS.items():
        pix = d[page - 1].get_pixmap(dpi=300, clip=fitz.Rect(*box))
        if name in BLANK:
            x0, y0, x1, y1 = BLANK[name]
            k = 300 / 72
            pix.set_rect(fitz.IRect(pix.x + int((x0 - box[0]) * k), pix.y + int((y0 - box[1]) * k),
                                    pix.x + int((x1 - box[0]) * k), pix.y + int((y1 - box[1]) * k)),
                         (255, 255, 255))
        out = os.path.join(HERE, 'panels', f'{name}.png')
        pix.save(out)
        print('wrote', out)


if __name__ == '__main__':
    main(sys.argv[1])
