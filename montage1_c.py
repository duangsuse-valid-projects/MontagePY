#!/bin/env python3
# -*- coding: utf-8 -*-

from numpy import array
from PIL import Image, ImageDraw, ImageFont
from cv2 import UMat, VideoCapture, VideoWriter
import cv2

from itertools import cycle
from montage1 import *

def solveItemColorLayout(img, item_size, scale, spacing):
  (width, height) = img.size
  (w_item, h_item) = tuple(int((item_size+sp)*scale) for sp in spacing)
  (m_item, n_item) = tuple(int(v) for v in [width / w_item, height / h_item])
  (padLeft, padTop) = tuple(int(v*scale / 2) for v in [(width % w_item), (height % h_item)])

  img_average = img.resize((m_item, n_item), Image.BICUBIC, box=(padLeft, padTop, img.width-padLeft, img.height-padTop))

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

def isColorNearTo(key_color, key_thres, color):
  diff = map(lambda c: abs(c[0] - c[1]), zip(color, key_color) )
  return sum(diff) < key_thres

# font, font_size, scale, spacing; key_color
def montage(image, cfg, calc_draw_color):
  newSize = tuple(int(d*cfg.scale) for d in image.size)
  scaledImage = image.resize(newSize, Image.ANTIALIAS) if cfg.scale != 1.0 else image
  newImage = Image.new(image.mode, newSize, cfg.key_color)
  areas = solveItemColorLayout(scaledImage, cfg.font_size, cfg.scale, cfg.spacing)
  drawTextMontage(newImage, areas, cycle(cfg.text), cfg.font, calc_draw_color)
  return newImage

def cvMontage(mat, cfg, calc_draw_color) -> array:
  img = Image.fromarray(array(mat))
  return array(montage(img, cfg, calc_draw_color))

def zipWithNext(xs: list):
  require(len(xs) % 2, lambda it: it == 0, "list not paired, rest ")
  for i in range(1, len(xs), 2):
    yield (xs[i-1], xs[i])

def expandRangeStarts(starts, n):
  indexed = list(range(n))
  def assign(start, stop, value):
    nonlocal indexed
    for i in range(start, stop): indexed[i] = value
  sorted_starts = sorted(starts, key=lambda it: it[0])
  for (a, b) in zipWithNext(sorted_starts):
    assign(a[0], b[0], a[1])
  (last, last_value) = sorted_starts[-1]
  assign(last, n, last_value)

def fileExtNameSplit(path):
  extIndex = path.rfind('.')
  return (path[:extIndex], path[extIndex+1:])

def playCvMontage(cap, cfg, calc_draw_color, title="Montage", filename="mon.avi"):
  (fps, width, height) = cv2VideoInfo(cap)
  print(f"{fps} {width}x{height}")
  vid = VideoWriter(filename, VideoWriter.fourcc(*"MJPG"), fps, (width,height))

  cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
  unfinished, img = cap.read()
  while unfinished:
    mon = cvMontage(img, cfg, calc_draw_color)
    cv2.imshow(title, mon)
    vid.write(mon)
    key = chr(cv2.waitKey(1) & 0xFF)
    if key == 'q': break
    unfinished, img = cap.read()
  vid.release()

def cv2VideoInfo(cap):
  props = [cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT]
  return tuple(int(cap.get(p)) for p in props)

def main(args):
  cfg = app.parse_args(args)
  cfg.font = ImageFont.truetype(cfg.font, cfg.font_size) if cfg.font != None else ImageFont.load_default()
  cfg.key_color = colorFromHtml(cfg.key_color)
  print(f"{cfg.font_size}px, {cfg.key_color} ±{cfg.key_thres} {cfg.spacing}")
  calc_draw_color = lambda c: None if isColorNearTo(cfg.key_color, cfg.key_thres, c) else c
  for path in cfg.images:
    (name, ext) = fileExtNameSplit(path)
    if ext in "mp4 webm mkv".split(" "):
      cap = VideoCapture(path)
      playCvMontage(cap, cfg, calc_draw_color, filename=f"{name}_mon.avi")
      cap.release()
    else:
      image = Image.open(path)
      montage(image, cfg, calc_draw_color).save(f"{name}_mon.png")

from sys import argv
if __name__ == "__main__": main(argv[1:])
