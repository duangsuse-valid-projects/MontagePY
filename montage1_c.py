#!/bin/env python3
# -*- coding: utf-8 -*-

from numpy import array
from PIL import Image, ImageDraw, ImageFont
from cv2 import UMat, VideoCapture, VideoWriter
import cv2

import srt

from itertools import cycle, repeat
from montage1 import *

def solveItemLayout(size, item_size, scale, spacing):
  (width, height) = size
  (w_item, h_item) = tuple((sz+sp)*scale for (sz, sp) in zip(item_size, spacing))
  (m_item, n_item) = tuple(int(v) for v in [width / w_item, height / h_item])
  (padLeft, padTop) = tuple(int(sz*scale / 4) for sz in [(width % w_item), (height % h_item)])
  return (w_item, h_item, m_item, n_item, padLeft, padTop)

def solveItemColors(img, layout):
  (width, height) = img.size
  (w_item, h_item, m_item, n_item, padLeft, padTop) = layout
  img_average = img.resize((m_item, n_item), Image.BICUBIC, box=(padLeft, padTop, img.width-padLeft, img.height-padTop))

  for i in range(0, n_item):
    for j in range(0, m_item):
      (y, x) = (padTop + i*h_item, padLeft + j*w_item)
      yield (x, y, img_average.getpixel((j, i)) )

def drawTextMontage(img, areas, seq, font, calc_draw_color):
  draw = ImageDraw.Draw(img)
  for (x, y, color) in areas:
    drawc = calc_draw_color(color)
    if drawc != None:
      draw.text((x, y), next(seq), font=font, fill=(drawc))

# util funcs
def isColorNearTo(key_color, key_thres, color):
  diff = map(lambda c: abs(c[0] - c[1]), zip(color, key_color) )
  return sum(diff) < key_thres

def mapUMatWithPillow(mat:UMat, transform) -> UMat:
  img = Image.fromarray(array(mat))
  return UMat(array(transform(img)))

def expandSrts(srts, fps, count, placeholder):
  indexed = [placeholder for _ in range(count)]
  no = lambda t: int(t.total_seconds() * fps)
  for srt in srts:
    start, end = no(srt.start), no(srt.end)
    indexed[start:end] = repeat(srt.content, end - start)
  return indexed

def cv2VideoInfo(cap):
  props = [cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT]
  return tuple(int(cap.get(p)) for p in props)

# font, font_size, scale, spacing; key_color
class Montage:
  def __init__(self, cfg, size):
    self.font = cfg.font; self.scale = cfg.scale; self.spacing = cfg.spacing
    self.text = cfg.text
    self.key_color = cfg.key_color; self.calc_draw_color = cfg.calc_draw_color
    self.back_color = let(colorFromHtml, cfg.mon_background)

    self.newSize = tuple(int(sz*cfg.scale) for sz in size)
    self.refreshLayout()
  def refreshLayout(self):
    if len(self.text) == 0: return
    self.layout = solveItemLayout(self.newSize, self.font.getsize(self.text[0]), self.scale, self.spacing)

  def runOn(self, image):
    areas = solveItemColors(image, self.layout)
    newImage = Image.new(image.mode, self.newSize, self.back_color or self.key_color)
    drawTextMontage(newImage, areas, cycle(self.text), self.font, self.calc_draw_color)
    return newImage

from time import time, strftime

def playCvMontage(cap, mon, title="Montage", filename="mon.avi", subtitle=None, placeholder="#"):
  (fps, count, _, _) = cv2VideoInfo(cap)
  vid = VideoWriter(filename, VideoWriter.fourcc(*"FMP4"), fps, mon.newSize)
  ary = expandSrts(subtitle, fps, count, placeholder) if subtitle != None else None

  cv2.namedWindow(title, cv2.WINDOW_NORMAL)
  begin_time = time()
  index = 0
  unfinished, img = cap.read()
  while unfinished:
    if subtitle != None:
      text0 = mon.text
      mon.text = ary[index]
      if mon.text != text0: mon.refreshLayout()
    mon_img = mapUMatWithPillow(img, mon.runOn)
    cv2.imshow(title, mon_img)
    vid.write(mon_img)

    key = chr(cv2.waitKey(1) & 0xFF)
    if key == 'q': break
    elif key == 'p':
      duration = time() - begin_time
      print("%i time=%.3fs %.3ffps" %(index, duration, index/duration) )
    unfinished, img = cap.read()
    index += 1
  vid.release()


from argparse import FileType

def fileExtNameSplit(path):
  extIndex = path.rfind('.')
  return (path[:extIndex], path[extIndex+1:])

def main(args):
  apg1.add_argument("--subtitle", type=FileType("r"), help="subtitle file for -text")
  apg1.add_argument("--subtitle-placeholder", type=str, default="#", help="placeholder for subtitle")
  apg1.add_argument("--mon-background", type=str, default=None, help="replacement back-color for mon (default -key-color)")
  readSrt = lambda it: srt.parse(it.read())

  cfg = app.parse_args(args)
  cfg.font = ImageFont.truetype(cfg.font, cfg.font_size) if cfg.font != None else ImageFont.load_default()
  cfg.key_color = colorFromHtml(cfg.key_color)

  print(f"{cfg.font_size}px, {cfg.key_color} ±{cfg.key_thres} {cfg.spacing}")
  cfg.calc_draw_color = lambda c: None if isColorNearTo(cfg.key_color, cfg.key_thres, c) else c
  for path in cfg.images:
    (name, ext) = fileExtNameSplit(path)
    if ext in "mp4 webm mkv flv".split(" "):
      cap = VideoCapture(path)
      (fps, count, width, height) = cv2VideoInfo(cap)
      print(f"{fps}fps*{count} {width}x{height}")

      mon = Montage(cfg, (width, height) )
      playCvMontage(cap, mon, filename=f"{name}_mon.avi", subtitle=let(readSrt, cfg.subtitle), placeholder=cfg.subtitle_placeholder)
      cap.release()
    else:
      image = Image.open(path)
      mon = Montage(cfg, image.size)
      mon.runOn(image).save(f"{name}_mon.png")

from sys import argv
if __name__ == "__main__": main(argv[1:])
