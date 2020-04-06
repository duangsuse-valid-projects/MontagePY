#!/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image, ImageDraw, ImageFont
from itertools import cycle

from montage1 import *

def solveItemColorLayout(img, item_size, scale, spacing):
  (width, height) = img.size
  (w_item, h_item) = tuple(int((item_size+sp)*scale) for sp in spacing)
  (m_item, n_item) = tuple(int(v) for v in [width / w_item, height / h_item])
  (padLeft, padTop) = tuple(int(v*scale / 2) for v in [(width % w_item), (height % h_item)])

  imgable_area = img.crop((padLeft, padTop, img.width-padLeft, img.height-padTop))
  img_average = imgable_area.resize((m_item, n_item), Image.ANTIALIAS)

  for i in range(0, n_item):
    for j in range(0, m_item):
      (y, x) = (i*h_item, j*w_item)
      yield (x, y, img_average.getpixel((j, i)) )

def drawTextMontage(img, areas, seq, font, calc_draw_color):
  draw = ImageDraw.Draw(img)
  for (x, y, color) in areas:
    drawc = calc_draw_color(color)
    if drawc != None:
      draw.text((x, y), next(seq), font=font, fill=colorBackHtml(drawc))

def averageColorUnlessIsBackground(color, key_color, key_thres):
  diff = map(lambda c: abs(c[0] - c[1]), zip(color, key_color) )
  return color if not sum(diff) < key_thres else None

def main(args):
  cfg = app.parse_args(args)
  font = ImageFont.truetype(cfg.font, cfg.font_size) if cfg.font != None else ImageFont.load_default()
  key_color = colorFromHtml(cfg.key_color)
  print(f"{cfg.font_size}px, {key_color} ±{cfg.key_thres}")
  calc_draw_color = lambda c: averageColorUnlessIsBackground(c, key_color, cfg.key_thres)
  for path in cfg.images:
    image = Image.open(path)
    newSize = tuple(int(d*cfg.scale) for d in image.size)
    scaledImage = image.resize(newSize, Image.ANTIALIAS) if cfg.scale != 1.0 else image
    newImage = Image.new(image.mode, newSize, key_color)
    areas = solveItemColorLayout(scaledImage, cfg.font_size, cfg.scale, cfg.spacing)
    drawTextMontage(newImage, areas, cycle(cfg.text), font, calc_draw_color)
    newImage.save(f"{path[:path.rfind('.')]}_mon.png")

from sys import argv
if __name__ == "__main__": main(argv[1:])
