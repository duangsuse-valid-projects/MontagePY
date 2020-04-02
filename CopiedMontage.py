#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple
from PIL import Image, ImageDraw, ImageFont


def rgbColorFrom(hexc):
  text = hexc.lstrip("#")
  return tuple([int(text[i:i+2], 16) for i in range(0, 6, 2)])
def rgbColorBack(color):
  hex_only = lambda i: hex(i)[2:].rjust(2,"0") #0x...
  return f"#{''.join(map(hex_only, color))}"

class Fold:
  def __init__(self): pass
  def accept(self, value): pass
  def finish(self): pass

class MapFold(Fold):
  def __init__(self, fold, n):
    self.folds = [fold() for _ in range(n)]
  def accept(self, value):
    for (f, v) in zip(self.folds, value): f.accept(v)
  def finish(self):
    return (it.finish() for it in self.folds)

class Averager(Fold):
  def __init__(self):
    self.n, self.k = 0, 0
  def accept(self, value):
    self.n += value
    self.k += 1
  def finish(self):
    return self.n / self.k

def averageColor(img, keyc) -> Tuple[tuple, int]:
  def dist(a, b): return abs(a - b) #< color distance
  averager = MapFold(Averager, Image.getmodebands(img.mode))
  n_thres = 0
  for y in range(0, img.height):
    for x in range(0, img.width):
      rgb = img.getpixel((x, y))
      averager.accept(rgb)
      dists = map(lambda ab: dist(ab[0], ab[1]), zip(keyc.color, rgb) )
      if sum(dists) > keyc.thres: n_thres += 1
  return (averager.finish(), n_thres)


class MontageConfig:
  def __init__(self, font, font_size, font_color, color_mode, spacing, drawcolor_not):
    self.font, self.font_size, self.font_color = font, font_size, font_color
    self.color_mode, self.spaceing = color_mode, spacing
    self.drawcolor_not = drawcolor_not

class KeyColor:
  def __init__(self, color, thres, take):
    self.color, self.thres, self.take = color, thres, take

def averageColor(img, keyc) -> Tuple[tuple, int]:
  def dist(a, b): return abs(a - b) #< color distance
  averager = MapFold(Averager, Image.getmodebands(img.mode))
  n_thres = 0
  for y in range(0, img.height):
    for x in range(0, img.width):
      rgb = img.getpixel((x, y))
      averager.accept(rgb)
      dists = map(lambda ab: dist(ab[0], ab[1]), zip(keyc.color, rgb) )
      if sum(dists) > keyc.thres: n_thres += 1
  return (averager.finish(), n_thres)

def getAvgColor(section, keyc) -> Tuple[tuple, bool]:
  pcounts = section.width * section.height
  if pcounts == 0: return (0, 0, 0), False
  (avgc, n_thres) = averageColor(section, keyc)
  return map(int, avgc), keyc.take(n_thres, pcounts)

font = ImageFont.load_default()
cfg = MontageConfig(font, 10, None, "RGB", (2,2), lambda it: getAvgColor(it, keyc) )
keyc = KeyColor(rgbColorFrom("#FFFFFF"), 10, lambda a, b: a > b * 0.5)

text = u"EmmM"
image = Image.open("pic.jpg")

scale = 1.0
new_coord = (it*scale for it in (image.width, image.height))
ims = image.resize(new_coord, Image.ANTIALIAS)
ims = ims.convert("RGB")

(width, height) = (ims.width, ims.height)
(h_sp, v_sp) = cfg.spaceing

(w_item, h_item) = (int((cfg.font_size+sp)*scale) for sp in (h_sp, v_sp) )
padLeft = int((width % w_item + h_sp) /2)
padTop = int((height % h_item + v_sp) /2)

im = Image.new("RGB", (width, height), keyc.color)
draw = ImageDraw.Draw(im)

from itertools import cycle
chars = cycle(text)
print(f"Generating... image size: {ims.width}x{ims.height}")

def divisibleRange(start, stop, step):
  return range(start, stop - (stop%step), step)

cfg.font_color = rgbColorBack(cfg.font_color) if cfg.font_color != None else None
for y in divisibleRange(padTop, height, h_item):
  for x in divisibleRange(padLeft, width, w_item):
    area = (x, y, x+w_item, y+h_item)
    shadow = image.crop(area)
    avgc, is_draw = getAvgColor(shadow, keyc)
    if is_draw:
      draw.text(area[0:2], next(chars), font=font, fill = cfg.font_color or rgbColorBack(avgc))

print("Finished!")
im.show()
