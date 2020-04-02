#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image, ImageDraw, ImageFont
from argparse import ArgumentParser, FileType

def rgbColorFromHex(hexc):
  hex_text = hexc.lstrip("#")
  return (int(hex_text[i:i+2], 16) for i in range(0, 6, 2))
def rgbColorBackHex(rgbc):
  joined = "".join(map(lambda i: hex(i)[2:].rjust(2,"0"), rgbc)) #0x...
  return f"#{joined}"


def divisableRange(start, stop, step):
  return range(start, stop - (stop%step), step)

def namedEval(expr, filename, **kwargs):
  return eval(compile(expr, filename, "eval", **kwargs))

def let(transform, self):
  return transform(self) if self != None else None

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

def imagePixels(img):
  for y in range(0, img.height):
    for x in range(0, img.width):
      yield img.getpixel((x, y))


app = ArgumentParser("montage", description="Create montage pictures(drawn with patterned elements)")

app.add_argument("-font", type=FileType("r"), default=None, help="otf/ttf font file path")
app.add_argument("-font-size", type=int, default=10, help="font size (e.g. 14)")
app.add_argument("-font-color", type=rgbColorFromHex, default=None, help="enforce font HTML color (if want)")

app.add_argument("-scale", type=float, default=1.0, help="1:1 scale for output image")
app.add_argument("-spacing", metavar=":h,v", type=str, default="0,0", help="horizontal,vertical spacing for items")

_color_code = "averageColorIfDifferThreshold(it, (0xFF,0xFF,0xFF))"
_image_code = "Image.new(it.mode, it.size)"
app.add_argument("--draw-color-code", type=str, default=_color_code, help="Python code(it: input Image) for item color")
app.add_argument("--new-image-code", type=str, default=_image_code, help="Python code(it: original Image) for new img creating")

app.add_argument("image", type=FileType("r"), help="image to draw with")
app.add_argument("-seq", type=str, default="Emmm", help="sequence of char to embed in")


def averageColor(img, key_color, thres):
  def dist(a, b): return abs(a - b) #< color distance
  averager = MapFold(Averager, Image.getmodebands(img.mode))
  n_inthres = 0
  for pix in imagePixels(img):
    averager.accept(pix)
    dists = map(lambda ab: dist(ab[0], ab[1]), zip(key_color, pix) )
    if sum(dists) < thres: n_inthres += 1
  return (averager.finish(), n_inthres)

def averageColorIfDifferThreshold(img, key_color, thres = 15, ratio=0.5):
  n_pixel = img.width * img.height
  if n_pixel == 0: return None
  (avgc, n_inthres) = averageColor(img, key_color, thres)
  return map(int, avgc) if n_inthres/n_pixel <= ratio else None


def solveLayout(size, font_size, scale, spacing):
  (width, height) = size
  (h_sp, v_sp) = spacing
  (w_item, h_item) = (int((font_size + sp) * scale) for sp in (h_sp, v_sp))
  padLeft = (width % w_item + h_sp) // 2
  padTop = (height % h_item + v_sp) // 2
  for y in divisableRange(padTop, height, h_item):
    for x in divisableRange(padLeft, width, w_item):
      yield (x, y, x+w_item, y+h_item)

def drawMontage(img, dest_img, areas, seq, font, draw_color, force_color=None):
  draw = ImageDraw.Draw(dest_img)
  for area in areas:
    shadowed = img.crop(area)
    fill = let(rgbColorBackHex, draw_color(shadowed))
    if fill != None:
      draw.text(area[0:2], next(seq), font=font, fill= force_color or fill)

from itertools import cycle

def cloneImageColored(img, bg_color):
  return Image.new(img.mode, img.size, bg_color)

def main(args):
  cfg = app.parse_args(args)
  font = let(lambda fp: ImageFont.truetype(fp.name, cfg.font_size), cfg.font) or ImageFont.load_default()
  font_color = let(rgbColorBackHex, cfg.font_color)

  image = Image.open(cfg.image.name)
  if cfg.scale != 1.0:
    new_coord = (int(it*cfg.scale) for it in image.size)
    image = image.resize(new_coord, Image.ANTIALIAS)
  #image = image.convert("RGB")

  spacing = map(int, cfg.spacing.lstrip(":").split(","))
  areas = solveLayout(image.size, cfg.font_size, cfg.scale, spacing)
  new_image = namedEval(f"lambda it, cfg: {cfg.new_image_code}", "<new_image_code>")(image, cfg)
  draw_color = namedEval(f"lambda it: {cfg.draw_color_code}", "<draw_color_code>")
  drawMontage(image, new_image, areas, cycle(cfg.seq), font, draw_color, force_color=font_color)

  new_image.show()

from sys import argv
if __name__ == "__main__": main(argv[1:])
