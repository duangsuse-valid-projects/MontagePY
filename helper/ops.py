# -*- coding: utf-8 -*-

''' Arbitrary operators '''

class Iinfty (int):
  ''' Infinity with sign (+/-) '''
  def __init__(self, posign=False): self.neg = posign
  def __abs__(self): return Iinfty(False)
  def __neg__(self): return Iinfty(not self.neg)
  def __bool__(self): return True
  def __add__(self, v): return self
  def __sub__(self, v):
    assert self.notrealeq(v), f'{self}-({v}) is undefined number'
    return self
  def __rsub__(self, v): return Iinfty(True)
  def __rmul__(self, v): return self.__mul__(v)
  def __rtruediv__(self, v): return self.changesign(v)
  def __rmod__(self, v): return v
  # +/+=+; +/-,-/+=-; -/-=+
  def changesign(self, v):
    return Iinfty(False if (v >=0 and not self.neg) or (v<0 and self.neg) else True)
  def notrealeq(self, n): return not (type(n) is Iinfty and n.neg==self.neg)
  def __mul__(self, v): return self.changesign(v)
  def __truediv__(self, v): return self.changesign(v)
  def __mod__(self, v): return Iinfty(v <0)
  def __divmod__(self, v): return (self.__truediv__(v), self.__mod__(v))
  def __gt__(self, n): return self.notrealeq(n) and not self.__lt__(n)
  def __ge__(self, n): return self.__gt__(n) or False#eq
  def __lt__(self, n):
    otherinf = (type(n) is not Iinfty); nneg = n <0
    if self.neg:
      return (True if not otherinf or (otherinf and not nneg) else False)
    else:
      return (False if not otherinf or (otherinf and not nneg) else True)
  def __le__(self, n): return self.__lt__(n) or False#eq
  def __eq__(self, other): return False
  def __ne__(self, other): return not self.__eq__(other)
  def __hash__(self): assert False, 'Infinity cannot have hashcode (they never == anything)'
  def signrep(self): return ('-' if self.neg else "")
  def __str__(self): return self.signrep()+'Infinity'
  __repr__ = __str__
  def __float__(self): return float('-inf' if self.neg else 'inf')

from .auxfun import compose, curry
from .auxfun import op_prefix, op_infix, op_postfix
from .auxfun import notp, TODO

zeroq = op_postfix('==0'); nzeroq = notp(zeroq)
emptyq = compose(zeroq, len); nempetyq = notp(emptyq)
noneq = op_postfix('is None'); nnoneq = notp(noneq)
modn = curry(op_infix('%')); mod2 = modn(2)

evenq = compose(zeroq, mod2)
oddq = notp(evenq)

nadd, nsub, nmul, ndiv = map(op_infix, '+,-,*,/'.split(','))
iadd, isub, imul, idiv = map(lambda f: lambda a,b: int(f(a,b)), [nadd, nsub, nmul, ndiv])

# distance, multiDistance, sumDistance
# ncmpLT, mdcmpLT
# mdadd, mdsub, ...
ncmpLT = op_infix('<')
# distanceBoundq(x0, maxd, distancef=..., cmpf=...)
# readAlignList, showAlignList, writeByteArray

class Averager:
  ''' Average value calculation tool '''
  def __init__(self):
    self.count = 0 #len
    self.total = 0 #sum
  def update(self, x, sumf=nadd, n=1):
    self.count += n
    self.total = sumf(self.total, x)
  def updateAll(self, xs, sumf=sum, lenf=len):
    self.count += lenf(xs)
    self.total += sumf(xs)
  def solve(self) -> float: return self.total / self.count
  MinMax: type

class _AvgMinMax (Averager):
  ''' With min/max value support '''
  def __init__(self, mn=0, mx=0):
    self.min = mn; self.max = mx
  def updateMinMax(self, lessThan, xm, xx):
    if lessThan(xm, self.min): self.min = xm
    elif lessThan(self.max, xx): self.max = xx
  def update(self, x, sumf=nadd, n=1, lessThan=ncmpLT):
    self.updateMinMax(lessThan, x, x)
    self.total = sumf(self.total, x)
    self.count += n
  def updateAll(self, xs, sumf=sum, lenf=len, minf=min, maxf=max, lessThan=ncmpLT):
    self.updateMinMax(lessThan, minf(xs), maxf(xs))
    self.total += sumf(xs)
    self.count += lenf(xs)
  def solve(self): return self.total / self.count
  def solve_ncorner(self, corners='_^'): TODO()
  def _minError(self): TODO()
  def _maxError(self): TODO()
  minError, maxError = property(_minError), property(_maxError)
Averager.MinMax = _AvgMinMax
