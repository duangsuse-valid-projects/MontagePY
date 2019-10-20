#!python3 -qi
# -*- coding: utf-8 -*-

'''
Simple parser combinator module

Abstraction: MarkReset, BaseParser, PsP(Parser combinator protocol), Parser

Item combinator: satisfy, char, elem, charseq
Group combinator: seq, br, repeat, oneOrMore, zeroOrMore
Predefined: wsP, ws0P

Helper:
  - MarkReset: mark/reset, with psr.positional():
  - BaseParser: consume(), issue(reason), eissue(fmtf, a0, a1), postinit()
  - BaseParser.ParserError: descriptor(), describe()
  - PsP: Matched, Unmatched, matchedq; identity, listfold

author: duangsuse
date: 2019-10-20
version: 1
'''

from os import linesep as lineSeprator
from math import isinf as isInfinity
from typing import Callable, TypeVar

T = TypeVar('T')

class MarkReset:
  ''' A class saving some state and thus it's changes can be reverted
    \nMultiply mark is supported by seprate implemenetation
    \nmark: save state; reset: revert '''
  def mark(self): pass
  def reset(self): pass
  class PositionalTask:
    ''' Given a MarkReset instance, provides with: syntax support '''
    def __init__(self, mr):
      self.mr = mr
    def __enter__(self):
      self.mr.mark()
    def __exit__(self, _w,_t,_f):
      self.mr.reset()
  def positional(self): return MarkReset.PositionalTask(self)

class Iterator:
  ''' Iterator providing [next(...) throws StopIteration] access '''
  def __next__(self): pass
  def __iter__(self): return self
class IteratorHelper:
  ''' Abstract class providing i++; i = i%len '''
  def __init__(self): self.lst=[]; self.i=0
  def moveNext(self):
    ''' return self.i++ '''
    oldi = self.i; self.i += 1
    return oldi
  def recycle(self):
    ''' i = i `mod` len(xs) '''
    self.i = self.i % len(self.lst)
  @staticmethod
  def clampIndex(i, size):
    return (0) if i<0 else min(size-1, i)
  @staticmethod
  def stop(): raise StopIteration()

class MarkResetListIterator (MarkReset, Iterator, IteratorHelper):
  def __init__(self, lst: list, i0=0):
    self.lst = lst; self.i = i0
    self.stack = []
  def __next__(self):
    nidx = self.moveNext()
    if nidx == len(self.lst): self.stop()
    try:
      return self.lst[nidx]
    except IndexError:
      self.i = len(self.lst)
      self.stop()
  def __length_hint__(self):
    return len(self.lst)
  def mark(self):
    self.stack.append(self.i)
  def reset(self):
    self.i = self.stack.pop()
  def peek(self): return self.lst[self.i]

class PsP:
  ''' Parser combinator protocol (Matched/Unmatched/matchedq); listfold; Infinity; answer '''
  Matched = lambda r: (True, r)
  Unmatched = (False, )
  matchedq = lambda r: len(r) == 2
  @staticmethod
  def also(obj, x, op): op(obj, x); return obj
  listfold = lambda l, x: PsP.also(l, x, lambda lst, t: lst.append(t[1]))
  Infinity = float('inf')
  answer = lambda pr, x: PsP.Matched(Func.delazy(x)) if pr else PsP.Unmatched

class Func:
  idp = lambda x: x
  delazy = lambda r_op: r_op() if callable(r_op) else r_op
  delazy1 = lambda r_f, x: r_f(x) if callable(r_f) else r_f
  linker = lambda nam: globals()[nam]

def Try(op, err=lambda e: lambda c: False, ok=lambda r: lambda c: c(r), exc=Exception):
  try: return Func.delazy1(ok, op())
  except exc as ex: return Func.delazy1(err, ex)
def cast(ty, x, checkf=lambda ty, x: type(x) is ty, castf=lambda ty, x: ty(x)):
  return x if checkf(ty,x) else castf(ty,x)

def hasName(o): return Try(lambda: o.__name__, False, True, exc=AttributeError)
def nameOf(o):
  try: return o.__name__
  except AttributeError: return o
def named(obj, nam):
  obj.__name__ = nam
  return obj
def clone(obj: T, mid='copy') -> T:
  #Try(lambda: obj.copy(), obj, exc=AttributeError)
  assert mid in dir(obj), f'Given object {obj} of type {type(obj)} is not cloneable'
  return obj.__getattribute__(mid)()

class ParserStream (MarkReset):
  ''' Simple char stream parser with (0) backtrace
      data (chars, fname,cnt, linecnt,colcnt, lineq, len=None)
      Markreset by self.chars
      use issue() to view error info
  '''
  def __init__(self, seq, fname='<string>', lineq=lambda c: c == lineSeprator, iterf=MarkResetListIterator):
    self.seq=seq; self.chars = iterf(seq); self.bakc0 = '\x00'
    self.fname=fname; self.cnt = 0; self.lastln = 0
    self.linecnt = 0; self.colcnt = 0; self.lineq = lineq
    self.len = None #full-readed stream
    self.stack = []
    self.postinit()
  def postinit(self): self.consume() #\x00
  def mark(self):
    self.stack.append((self.bakc0, self.cnt, self.lastln, self.linecnt, self.colcnt))
    self.chars.mark()
  def reset(self):
    self.bakc0, self.cnt, self.lastln, self.linecnt, self.colcnt = self.stack.pop()
    self.chars.reset()
  def finalyield(self):
    if self.len !=None:
      self.bakc0 = '\x00' #EOS
      raise StopIteration() #盖棺定论
    else: self.len = self.cnt
  def consume(self):
    ''' Move dataptr next; returning old character '''
    oldbakc0 = self.bakc0
    try: self.bakc0 = next(self.chars); self.cnt += 1; self.colcnt += 1
    except StopIteration: self.finalyield()
    if self.lineq(self.bakc0):
      self.linecnt += 1; self.colcnt = 0; self.lastln = self.cnt
    return oldbakc0
  def skip(self, n):
    ''' 因为时序逻辑关系有点混乱，所以不容易写对，凑合用
        注意：skip 时遇到 EOF 的最后一个 bakc0，会默认为已经跳过，这和 consme() 的行为一致 '''
    ch: str = '\x00'
    while n != 0:
      try: n -= 1; ch = next(self.chars); self.cnt += 1
      except StopIteration: self.finalyield()
      if self.lineq(ch):
        self.linecnt += 1; self.lastln = self.cnt
    self.colcnt = self.cnt - self.lastln
    self.bakc0 = ch
  class ParserError (Exception):
    ''' Parser combinator input parse failure (possible bad user-input) '''
    def __init__(self, p, r):
      self.filename = p.fname; self.pos = p.cnt
      self.lineno = p.linecnt; self.column = p.colcnt
      self.reason = r; self.line = p.thisline(); self.thisc = p.bakc0
    def descriptor(self): return (self.filename, self.lineno, self.column)
    def describe(self): return ':'.join(map(str, self.descriptor()))
    def __str__(self): return f'{self.describe()}: {self.reason}\n  {self.line} ({self.thisc})'
    __repr__ = __str__
  # Issue new ParserError with reason
  def issue(self, err='???'): return ParserStream.ParserError(self, err)
  # Throw new ParserError if fmtf is given, with 2 args
  def eissue(self, fmtf, ps, x):
    if fmtf !=None: raise self.issue(fmtf(ps, x))
  def thisline(self, vp=75, rpad=1): return self.seq[self.lastln:self.cnt-rpad][:vp]
  def view(self, vp=20, rmargin=0):
    ''' See the slice seq[cnt-1:cnt-1+vp], with clipped indices '''
    slen = len(self.seq)
    s = IteratorHelper.clampIndex(self.cnt -1-rmargin, slen)
    e = IteratorHelper.clampIndex(s+vp, slen)
    return self.seq[s:e]

class ParserKst:
  LstFold = (list(), PsP.listfold)
  CntFold = (0, lambda ac, _: ac+1)
  LenFold = (0, lambda ac, t: ac+len(t[1]))
  @staticmethod
  def HistFoldl(d, obj, cntf=lambda x: 1, initialf=lambda x: 0, mapf=lambda t: t[1]):
    x = mapf(obj)
    if not x in d: d[x] = initialf(x)
    d[x] += cntf(x)
    return d
  WS = named(lambda c: c.isspace(), 'ws')
  DIGIT = named(lambda c: c.isdigit(), 'digit')
  IDENTIFIER = named(lambda c: c.isidentifier(), 'ident')
  @staticmethod
  def join(xs, sep=':', exclf=lambda c: c.isspace(), reprf=str, usesepq=any):
    ''' if exists hasspace(x) in xs: sep.join else "".join '''
    xs = list(map(reprf, xs)) #immut
    hasspace = usesepq(map(exclf, xs))
    len0 = 1 if len(xs) ==0 else len(xs[0])
    uniqlen = list(map(len, xs)).count(len0) == len(xs)
    return (sep if hasspace or not uniqlen else "").join(xs)
  DFCharseq = lambda s, i: f'Expecting <{s[:i]}[{s[i:]}]>'
  DFSeq = lambda ps, i: f'Failed @{i}: {ps[i]}'
  DFBr = lambda ps, _: f'All {len(ps)} branch failed'
  DFRepeat = lambda pr, xs: f'Expecting {pr[1]} elements of {pr[0]}, {len(xs)} found'
  HistFold = ({}, HistFoldl)
ParserKst.HistFold = ({}, ParserKst.HistFoldl)

class Parser:
  ''' Extended parser combinator abstraction accept(ins) '''
  @staticmethod
  def run(parse, stm):
    ''' Run the parser with input stm, destruct input '''
    return parse.accept(cast(ParserStream, stm))
  @staticmethod
  def tryRun(parse, stm, exc=StopIteration):
    ''' Run parser, but consider StopIteration as Unmatched '''
    return Try(lambda: Parser.run(parse, stm), PsP.Unmatched, Func.idp, exc=exc)
  # Child-class abstraction
  def accept(self, ins: ParserStream): pass
  def __repr__(self): pass
  def __str__(self): return self.__repr__()
class Combined:
  ''' Combined(with its children) parser '''
  def __init__(self):
    ''' Registered subparsers(subs);
        Error handling formatter(errfmt)
        Parsed item reducer(fold) '''
    self.subs: list = []
    self.errfmt = print #joke!
    self.fold: (T, Callable[[T, T], T]) = ParserKst.LstFold
  def joinSubs(self, mid): return ParserKst.join(self.subs, sep=mid)

class satisfy (Parser):
  ''' item satisfys predicate? [p(x)] '''
  def __init__(self, p: callable, f: callable=Func.idp):
    self.p = p; self.f = f
  def accept(self, ins):
    return PsP.answer(self.p(ins.bakc0), lambda: self.f(ins.consume()))
  def __repr__(self):
    return f'({nameOf(self.p)}?)'
class char (Parser):
  ''' Full-equals parser A '''
  def __init__(self, x): self.item = x
  def accept(self, ins):
    eq = (ins.bakc0 == self.item)
    return PsP.answer(eq, lambda: ins.consume())
  def __repr__(self): return f'{self.item}'
class elem (Parser):
  ''' item element-of parser [...] '''
  def __init__(self, xset): self.xset = xset
  def accept(self, ins):
    inset = ins.bakc0 in self.xset
    return PsP.answer(inset, lambda: ins.consume())
  def __repr__(self): return f'[{ParserKst.join(self.xset)}]'

class charseq (Parser):
  ''' Item sequence parser A... '''
  def __init__(self, xs, errfmt=ParserKst.DFCharseq):
    self.xs = xs; self.errfmt = errfmt
  def accept(self, ins):
    for seqno, cha in enumerate(self.xs):
      if ins.bakc0 == cha: ins.consume()
      else:
        ins.eissue(self.errfmt, self.xs, seqno)
        return PsP.Unmatched
    return PsP.Matched(self.xs)
  def __repr__(self): return f'<{ParserKst.join(self.xs,sep="")}>'

class seq (Parser, Combined):
  ''' Sequence parser combinator A0.A1.A2 '''
  def __init__(self, *ps, fold=ParserKst.LstFold, errfmt=ParserKst.DFSeq):
    self.subs = ps; self.fold = fold; self.errfmt = errfmt
  def accept(self, ins):
    lhs = clone(self.fold[0])
    for i, p in enumerate(self.subs):
      rhs = Parser.tryRun(p, ins)
      if not PsP.matchedq(rhs):
        ins.eissue(self.errfmt, self.subs, i)
        return PsP.Unmatched
      lhs = self.fold[1](lhs, rhs)
    return PsP.Matched(lhs)
  def __repr__(self): return f'({self.joinSubs(" . ")})'

class br (Parser, Combined):
  ''' Branch of subparsers A|B|C, and A,B,C... can consume more than 1 char '''
  def __init__(self, *ps, errfmt=ParserKst.DFBr):
    self.subs = ps; self.errfmt = errfmt
    def removeWarn(sp): sp.errfmt = None
    for sub in self.subs: Try(lambda: removeWarn(sub))
  def accept(self, ins):
    for parser in self.subs:
      with ins.positional():
        res0 = Parser.run(parser, ins)
      if PsP.matchedq(res0): return PsP.Matched(res0)
    ins.eissue(self.errfmt, self.subs, None)
    return PsP.Unmatched
  def __repr__(self): return f'({self.joinSubs("|")})'

class repeat (Parser):
  ''' Repeat parser (x**n) '''
  def __init__(self, parse, atmin, atmax,
      fold=ParserKst.LstFold, errfmt=ParserKst.DFRepeat):
    assert atmin >= 0, f'Range start must be postitive (atmin={atmin})'
    assert atmax >= atmin, f'Range end must not be `lessThan start` ({atmin-atmax})'
    self.parse = parse; self.fold = fold; self.errfmt = errfmt
    self.min = atmin; self.max = atmax
  def accept(self, ins):
    lhs = clone(self.fold[0]); cnt = 0
    while cnt < self.max:
      rhs = Parser.tryRun(self.parse, ins)
      if PsP.matchedq(rhs):
        lhs = self.fold[1](lhs, rhs)
      elif cnt < self.min: #unmatched
        ins.eissue(self.errfmt, (self.parse, (self.min, self.max+1)), lhs)
        return PsP.Unmatched
      else: break
      cnt += 1 #inc count
    return PsP.Matched(lhs)
  def __repr__(self):
    if isInfinity(self.max) and self.min in [0,1]:
      postfix = ['*', '+'][self.min]
      return repr(self.parse)+postfix
    if hasName(self): return nameOf(self)
    return f'({self.parse} **[{self.min}...{self.max}])'

def optional(p): return named(repeat(p, 0, 1), f'[ {nameOf(p)} ]')
def oneOrMore(p, **kwargs): return repeat(p, 1, PsP.Infinity, **kwargs)
def zeroOrMore(p, **kwargs): return repeat(p, 0, PsP.Infinity, **kwargs)

singleWs = satisfy(ParserKst.WS)
wsP = oneOrMore(singleWs, fold=ParserKst.HistFold)
ws0P = zeroOrMore(singleWs, fold=ParserKst.HistFold)
