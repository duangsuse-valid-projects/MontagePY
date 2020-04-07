#!/bin/env python3
# -*- coding: utf-8 -*-

from numpy import array
from PIL import Image, ImageDraw, ImageFont
from cv2 import UMat, VideoCapture, VideoWriter
import cv2

from srt import Subtitle, parse

from itertools import cycle
from montage1 import *

def solveItemLayout(size, item_size, scale, spacing):
  (width, height) = size
  (w_item, h_item) = tuple((sz+sp)*scale for (sz, sp) in zip(item_size, spacing))
  (m_item, n_item) = tuple(int(v) for v in [width / w_item, height / h_item])
  (padLeft, padTop) = tuple(int(sz*scale / 2) for sz in [(width % w_item), (height % h_item)])
  return (w_item, h_item, m_item, n_item, padLeft, padTop)

def solveItemColors(img, layout):
  (width, height) = img.size
  (w_item, h_item, m_item, n_item, padLeft, padTop) = layout
  img_average = img.resize((m_item, n_item), Image.BICUBIC, box=(padLeft, padTop, img.width-padLeft, img.height-padTop))

  for i in range(0, n_item):
    for j in range(0, m_item):
      (y, x) = (padTop+ i*h_item, padLeft+ j*w_item)
      yield (x, y, img_average.getpixel((j, i)) )

def drawTextMontage(img, areas, seq, font, calc_draw_color):
  draw = ImageDraw.Draw(img)
  for (x, y, color) in areas:
    drawc = calc_draw_color(color)
    if drawc != None:
      draw.text((x, y), next(seq), font=font, fill=colorBackHtml(drawc))

# font, font_size, scale, spacing; key_color
def montage(image, cfg):
  newSize = tuple(int(d*cfg.scale) for d in image.size)
  scaledImage = image.resize(newSize, Image.BICUBIC) if cfg.scale != 1.0 else image
  layout = solveItemLayout(newSize, cfg.font.getsize(cfg.text[0]), cfg.scale, cfg.spacing)
  areas = solveItemColors(scaledImage, layout)
  newImage = Image.new(image.mode, newSize, cfg.key_color)
  drawTextMontage(newImage, areas, cycle(cfg.text), cfg.font, cfg.calc_draw_color)
  return newImage


def isColorNearTo(key_color, key_thres, color):
  diff = map(lambda c: abs(c[0] - c[1]), zip(color, key_color) )
  return sum(diff) < key_thres

def pillowCvify(transform, *args, **kwargs):
  def invoke(mat: UMat) -> UMat:
    img = Image.fromarray(array(mat))
    return UMat(array(transform(img, *args, **kwargs)))
  return invoke

def fileExtNameSplit(path):
  extIndex = path.rfind('.')
  return (path[:extIndex], path[extIndex+1:])

def cv2VideoInfo(cap):
  props = [cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT]
  return tuple(int(cap.get(p)) for p in props)


def playCvMontage(cap, cfg, title="Montage", filename="mon.avi"):
  (fps, width, height) = cv2VideoInfo(cap)
  print(f"{fps} {width}x{height}")
  vid = VideoWriter(filename, VideoWriter.fourcc(*"FMP4"), fps, (width,height))
  cvMontage = pillowCvify(montage, cfg)

  cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
  unfinished, img = cap.read()
  while unfinished:
    mon = cvMontage(img)
    cv2.imshow(title, mon)
    vid.write(mon)
    key = chr(cv2.waitKey(1) & 0xFF)
    if key == 'q': break
    unfinished, img = cap.read()
  vid.release()


def main(args):
  cfg = app.parse_args(args)
  cfg.font = ImageFont.truetype(cfg.font, cfg.font_size) if cfg.font != None else ImageFont.load_default()
  cfg.key_color = colorFromHtml(cfg.key_color)
  print(f"{cfg.font_size}px, {cfg.key_color} ±{cfg.key_thres} {cfg.spacing}")
  cfg.calc_draw_color = lambda c: None if isColorNearTo(cfg.key_color, cfg.key_thres, c) else c
  for path in cfg.images:
    (name, ext) = fileExtNameSplit(path)
    if ext in "mp4 webm mkv".split(" "):
      cap = VideoCapture(path)
      playCvMontage(cap, cfg, filename=f"{name}_mon.avi")
      cap.release()
    else:
      image = Image.open(path)
      montage(image, cfg).save(f"{name}_mon.png")

from sys import argv
if __name__ == "__main__": main(argv[1:])
