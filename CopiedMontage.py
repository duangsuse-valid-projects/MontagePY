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

text = u"T"
image = Image.open("pic.jpg")

font = ImageFont.load_default()
keyc = KeyColor(rgbColorFrom("#FFFFFF"), 10, lambda a, b: a > b * 0.5)
cfg = MontageConfig(font, 10, None, "RGB", (2,2), lambda it: getAvgColor(it, keyc) )

scale = 1.0
new_coord = (it*scale for it in (image.width, image.height))
ims = image.resize(new_coord, Image.ANTIALIAS)
ims = ims.convert("RGB")
(sp_h, sp_v) = cfg.spaceing
secWidth = int((sp_h + cfg.font_size)*scale)
secHeight = int((sp_v + cfg.font_size)*scale)
secRows = int(ims.height / secHeight)
secCols = int(ims.width / secWidth)
leftPadding = int((ims.width - secCols * secWidth) / 2) + int(sp_h / 2)
upPadding = int((ims.height - secRows * secHeight) / 2) + int(sp_v / 2)
im = Image.new("RGB", (ims.width, ims.height), keyc.color)
draw = ImageDraw.Draw(im)

from itertools import cycle
chars = cycle(text)
texty = upPadding
print(f"Generating... image size: {ims.width}x{ims.height}")
for i in range(0, secRows):
    textx = leftPadding
    for j in range(0, secCols):
        tempsec = ims.crop((textx, texty, textx + secWidth, texty + secHeight))
        textx = textx + secWidth
        avgc, is_draw = getAvgColor(tempsec, keyc)
        if not is_draw: continue
        if cfg.font_color != None:
            draw.text((textx, texty), next(chars), font=font, fill=cfg.font_color)
        else:
            draw.text((textx, texty), next(chars), font=font, fill=rgbColorBack(avgc))
    texty = texty + secHeight

print("Finished!")
im.show()
