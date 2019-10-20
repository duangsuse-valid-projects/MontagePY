# -*- coding: utf-8 -*-

''' Digital Image Processing Auxiliary '''

from .auxfun import TODO

class Range2D: # (Sized2D)
  def __init__(self, xrange, yrange):
    self.hline = xrange; self.vline = yrange
  def getsize(self): return (len(self.hline), len(self.vline))
  def sizerange(self): return self
  def box(self) -> (int, int, int, int):
    ''' (left, upper, right, lower) '''
    return (self.hline.start, self.vline.stop, self.hline.stop, self.vline.start)
  def edgeLD(self) -> (int,int): return (self.hline.start, self.vline.start)
  def edgeRT(self) -> (int,int): return (self.hline.stop, self.vline.stop)
  def _pA(self): return (self.hline.start, self.vline.stop)
  def _pB(self): return (self.hline.stop, self.vline.stop)
  def _pC(self): return (self.hline.start, self.vline.start)
  def _pD(self): return (self.hline.stop, self.vline.start)
  pA, pB, pC, pD = property(_pA), property(_pB), property(_pC), property(_pD)
  def _pCenter(self): return (len(self.hline)/2 +self.hline.start, len(self.vline)/2 +self.vline.start)
  pCenter = property(_pCenter)
  def slice(self, matr): TODO()
  def subimage(self, img): TODO()
  def points(self, ord='xy'): TODO()
  def edges(self, es='abcd'): TODO()
  def vertices(self, vs='1234'): TODO()
  def mids(self, vs='1234'): TODO()
  def scale(self, ratio): TODO()
  def crossLine(self, lns='/\\'): TODO()
  @staticmethod
  def from_psize(p, size):
    ''' P(x,y); Size(w,h) '''
    return Range2D(range(p[0], size[0]), range(p[1], size[1]))

class Sized2D:
  def getsize(self) -> (int, int): TODO()
  def sizerange(self) -> Range2D: TODO()
  
class Drawable:
  def draw(self, img, pos): TODO()
  
class TintedDrawable (Drawable):
  def tint(self, cv): TODO()
  def tintedDraw(self, img, pos, tint): TODO()
  def get_color(self): TODO()
  class TintedDrawTask:
    def __init__(self): TODO()
    def __enter__(self): TODO()
    def __exit__(self, w,t,f): TODO()
  def tinted(self) -> TintedDrawTask: TODO()
