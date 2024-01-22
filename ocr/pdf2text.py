import os
import csv
import subprocess

import sys, fitz
import os
import datetime
import argparse
import pytesseract
from PIL import Image


def get_args_parser_pdf2text():
    parser = argparse.ArgumentParser()

    parser.add_argument('--type', default='TCGA_BRCA', type=str, help='name in TCGAs')
    parser.add_argument('--root', default='/yanglin/GDC_DATA', type=str, help='path to TCGA')
    parser.add_argument('--savepath', default='/chenpingyi/projects/WSI-GPT/ocr/datasets', type=str,
                        help='path to wsi-text pairs')
    args, unparsed = parser.parse_known_args()

    for arg in vars(args):
        if vars(args)[arg] == 'True':
            vars(args)[arg] = True
        elif vars(args)[arg] == 'False':
            vars(args)[arg] = False

    return args


def find_path_pdf(root):
    slides = []
    for filepath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            path = os.path.join(filepath, filename)

            if path.endswith('PDF') or path.endswith('pdf'):  # key to find pdfs

                slides.append(path)

    return slides


def find_path_wsi(root):
    slides = []
    for filepath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            path = os.path.join(filepath, filename)

            if path.endswith('svs'):  # key to find wsis
                if 'DX1' in path:  # diagnostic slide
                    slides.append(path)
    return slides


def pyMuPDF_fitz(pdfPath, imagePath):  # pdf2img

    pdfDoc = fitz.open(pdfPath)
    pagecount = pdfDoc.page_count
    for pg in range(pdfDoc.page_count):
        page = pdfDoc[pg]
        rotate = int(0)

        # 每个尺寸的缩放系数为1.3，这将为我们生成分辨率提高2.6的图像。
        zoom_x = 1.3  # (1.33333333-->1056x816)   (2-->1584x1224)
        zoom_y = 1.3

        mat = fitz.Matrix(zoom_x, zoom_y).prerotate(rotate)

        pix = page.get_pixmap(matrix=mat, alpha=False)

        if not os.path.exists(imagePath):  # 判断存放图片的文件夹是否存在
            os.makedirs(imagePath)  # 若图片文件夹不存在就创建

        pix.save(imagePath + '/' + 'images_%s.png' % (pg + 1))  # 将图片写入指定的文件夹内

    return pagecount


def img2text(imgpath, pagecount, textpath):
    with open(textpath, "w") as f:
        pages = pagecount
        n = 0

        for n in range(pages):
            i = 0
            j = 0

            name = f'{imgpath}/images_' + str(n + 1) + '.png'
            image = Image.open(name)
            res = pytesseract.image_to_string(image, lang='eng')
            string_list = []
            for i in range(len(res)):
                for j in res[i]:
                    string_list.append(j)
            ocr_result_string = "".join(string_list)
            print("Page ", str(n + 1))
            print(ocr_result_string)
            f.write(ocr_result_string)  # 这句话自带文件关闭功能，不需要再写f.close()
            print("转换结束。")


def pdf2text(pdfpath, savepath, name):
    # pdfpath: path to pdf files
    # savepath: path to save pdf2img files(PNG) and img2text files(TXT)
    # name: the name for TXT file
    pagecount = pyMuPDF_fitz(pdfpath, savepath)
    img2text(savepath, pagecount, name)


if __name__ == "__main__":
    args = get_args_parser_pdf2text()
    root = os.path.join(args.root, args.type)
    pdfs = find_path_pdf(root)
    wsis = find_path_wsi(root)
    print(f'counts of wsis: {len(wsis)} , counts of pdfs: {len(pdfs)}')

    pairs = []
    for pdf in pdfs:
        prefix = pdf.split('/')[-1].split('.')[0]  # for exmplae ; 'TCGA-A8-A09K'

        wsiPath = [wsi for wsi in wsis if wsi.split('/')[-1].startswith(prefix)]

        if len(wsiPath) >= 2:
            DX = []

            for item in wsiPath:
                if item.split('/')[-1].split('-')[3][
                   :2] == '01':  # Barcode for TCGA. '01' means primariy solid tumor;'11' means solid tissue normal
                    DX.append(item)

        if len(wsiPath) >= 2:
            continue

        assert len(wsiPath) < 2, f'not found matched wsi for {pdf}, wsipath includes {wsiPath}'

        if len(wsiPath) == 0:
            continue
        else:
            pairs.append((pdf, wsiPath[0]))

    # save image-pdf-report pairs in individual directories named by slide-id
    for pdfPath, wsiPath in pairs:
        prefix = pdfPath.split('/')[-1].split('.')[0]

        savepath = os.path.join(args.savepath, args.type, prefix)

        pdf2text(pdfPath, savepath, os.path.join(savepath, 'Report.txt'))
        command = f"cp  {pdfPath} {savepath} " + "&" + f"cp -s {wsiPath} {savepath}"
        subprocess.Popen(command, shell=True)

    # collect reports into a txt
    for pdfPath, wsiPath in pairs:
        prefix = pdfPath.split('/')[-1].split('.')[0]

        savepath = os.path.join(args.savepath, args.type, prefix)

        with open(os.path.join(savepath, 'Report.txt'), "r", encoding='utf-8') as f:
            data = f.read()
            print(data)