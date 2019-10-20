#!/bin/python3 -qi
# -*- coding: utf-8 -*-

from typing import Iterable, Iterator
from typing import Any, Tuple, List, Dict
from typing import Generic, TypeVar, Callable

'''
Auxilary functions, like what defined in Haskell
Provides a well-__repr__ functional programming style library for Python3

- Lambdas: 入, curry, tcurry, tuncurry; delzy, delazy1, idp, ap
- Function operation: compose, flip
- Auxilary: Try, cast, Infinity
- Streams: baserange, chrange, distrange, repeat, concat, every
- Logical relations: andp, or_p, notp, nandp, xorp, allp, existp
- Operator notation: op_infix, op_prefix, op_postfix
- Misc operators: its, fst/snd, noneq, dentry, entries
'''

## Part 0.0: Lambda expressions TODO
def lambdaParse(expr: str) -> list:
  S0, St, Sb = range(0, 3)
  TODO()
def lambdaSubtract(l, tab) -> list: TODO()
def fmt_ap(lexp, atbl) -> str:
  ''' (\\x. x) x == x '''
  lam = lambdaParse(lexp)
  sexps = lambdaSubtract(lam, dict(zip(lam[0], atbl)))
  return ''.join(sexps)
  TODO()
def fmt_nameonlyf(nam, _): return nam

## Part 0: Lambdas
def nameOf(o, adesc=None, fmt=lambda ad: f" %({', '.join(map(str,ad))})"):
  name = o.__name__
  if len(name) ==0: return''
  if name[0] == '$': #execute
    colidx = name.find(':')
    cmd = name[1:colidx]; fname = name[colidx+1:]
    return globals()[cmd](fname, adesc)
  else: return name+(fmt(adesc) if adesc !=None else "")
def lambdaNamefmt(nam, bord='(%s)', cq=lambda c: c.isspace()):
  return (bord %nam if any(map(cq, nam)) else nam)
def 入(fn, name) -> callable:
  fn.__qualname__ = lambdaNamefmt(name)
  fn.__name__ = fn.__qualname__
  return fn

# curry :: ((a, b) -> r) -> (a -> (b -> r))
def curry(f) -> callable: return 入(lambda x0: 入(lambda x1: f(x0, x1),
  f'λx. {nameOf(f, (x0, "x"))}'), f'λx. {nameOf(f, ("x", "_"))}')
# tcurry: (( (a,b) ) r) ((a, b) r)
def tcurry(f) -> callable:
  return 入(lambda t0, t1: f((t0, t1)), f'{nameOf(f, ("x","y"))}ᵀ')
# tuncurry: ((a, b) r) (( (a,b) ) r)
def tuncurry(f) -> callable:
  return 入(lambda t: f(t[0], t[1]), f'{nameOf(f, ("x₀","x₁"))}')

# delazy: (() r | r) r
def delazy(r_op): return r_op() if callable(r_op) else r_op
入(delazy, 'λx.$x')
# delazy1: ((a) r | r) r
def delazy1(r_f, x): return r_f(x) if callable(r_f) else r_f
入(delazy1, 'λrf. rf(x) || rf')
# idp :: a -> a
def idp(x): return x
入(idp, 'λx.x')
def ap(f): return f()
入(ap, 'λf.f()')

## Part 1: Functional operators
def compose(g, f):
  ''' compose :: (b -> c) -> (a -> b) -> (a -> c) '''
  return 入(lambda x: g(f(x)), f'λx. {nameOf(f, ("x"))} » {nameOf(g, ("x"))}')
def pipe(*fs): return foldr(compose, idp)(fs)
def fmt_flipf(nam, atbl): return '`%s(%s)' %(nam, ', '.join(map(str, reversed(atbl or []))))
def flip(f):
  ''' flip :: (a -> b -> c) -> (b -> a -> c) '''
  return 入(lambda x0, x1: f(x1, x0), f'$fmt_flipf:{nameOf(f)}')

## Part 2: Auxilary
def Try(op, fail=lambda e: lambda c: False, normal=lambda r: lambda c: c(r), err=Exception):
  try:
    return delazy1(normal, op())
  except err as ex:
    return delazy1(fail, ex)
def cast(ty, o, castf=lambda ty, x: ty(x), inschkf=lambda ty, x: type(x) is ty):
  return o if inschkf(ty,o) else castf(ty,o)
Infinity = cast(float, 'inf')

E, T, K, V = TypeVar('E'), TypeVar('T'), TypeVar('K'), TypeVar('V')

## Part 3: Streams
class baserange:
  ''' A base class for custom range, delegate for built-in range
      Provides: rng :: (s, e, k)
      Instance: List {len, getitem, iter, reversed, contains} / Obj {init, eq, hash, str}
      Left-inclusive & Right-exclusive; range('x', 'z') should be [x, y] (len=2)
  '''
  def __init__(self, start, end, step=1):
    self.r = range(start, end, step)
  def __eq__(self, other): return self.r.__eq__(other.r)
  def __hash__(self): return self.r.__hash__()
  def __str__(self):
    rstep = self.r.step; simplew = rstep == 1
    return '(%s..%s%s)' %(self.r.start, self.r.stop, "" if simplew else f', {rstep}')
  __repr__ = __str__
  def __len__(self): return self.r.__len__()
  def __getitem__(self,i): return self.r.__getitem__(i)
  def __iter__(self): return self.r.__iter__()
  def __reversed__(self): return self.r.__reversed__()
  def __contains__(self,x): return self.r.__contains__(x)
  def rangeTuple(self): return (self.r.start, self.r.stop, self.r.step)
  rtuple = property(rangeTuple, doc='Tuple (start, end, step)')
class xxrange (baserange, Generic[E]):
  ''' WTF range implemented with (T -> Int) and (Int -> T) '''
  @classmethod
  def t2int(cls, x: E) -> int: TODO()
  @classmethod
  def int2t(cls, i: int) -> E: TODO()
  def __init__(self, start: E, end: E, step=1):
    super().__init__(self.t2int(start), self.t2int(end), step)
  def __getitem__(self,i): return self.int2t(super().__getitem__(i))
  def __iter__(self) -> Iterator[E]: return map(self.int2t, super().__iter__())
  def __reversed__(self): return map(self.int2t, super().__reversed__())
  def __contains__(self,x): return super().__contains__(self.t2int(x))
  def __repr__(self): return super().__str__()+self.describe()
  def describe(self): return '[%c-%c]' %tuple(map(self.int2t, self.rtuple[:-1]))

class chrange (xxrange[str]):
  t2int = ord; int2t = chr

#placeholder distange
def repeat(x, n=Infinity):
  assert n>=0, f'n({n}) must not be negative'
  while n !=0: n-=1; yield x # 我憎恨时序逻辑！
def concat(*args):
  for xs in args:
    for x in xs: yield x
def _foldr(op, r, lst):
  if len(lst) == 0: return r
  return op(lst[0], _foldr(op, r, lst[1:]))
def foldr(op, r): return 入(lambda xs: _foldr(op, r, xs), f'foldr{nameOf(op)}:{r}')
rassoc = foldr(入(lambda x, r: [x, r], 'λx rec. [x, rec]'), [])
every = curry(map)

## Part 4: Logical operators
global any, all

def logicfmt(f0, r, f1, surr='(%s)'): return surr %f'{nameOf(f0)}{r}{nameOf(f1)}'

def wtf(q, tf, ff): lambda x: (delazy(tf) if q(x) else delazy(ff))
def andp(q0, q1): return 入(lambda x: q0(x) and q1(x), logicfmt(q0, '∧', q1))
def or_p(q0, q1): return 入(lambda x: q0(x) or q1(x), logicfmt(q0, '∨', q1))
def notp(q): return 入(lambda x: not q(x), f'¬{nameOf(q)}')
def nandp(q0, q1): return 入(compose(notp, andp(q0, q1)), logicfmt(q0, '⊼', q1))
def norp(q0, q1): return 入(compose(notp, or_p(q0, q1)), logicfmt(q0, '⊽', q1))
def xorp(q0, q1): return 入(andp(nandp(q0, q1), or_p(q0, q1)), logicfmt(q0, '⊻', q1))

def allp(q): return 入(compose(all, every(q)), f'λxs. ∀x∈xs. {nameOf(q)}(x)')
def existp(q): return 入(compose(any, every(q)), f'λxs. ∃x∈xs. {nameOf(q)}(x)')

# filter: Callable[[Callable[[E], bool], Iterable[E]], Iterator[E]]
def gfilter(p: Callable[[E], bool], xs: Iterable[E]) -> Iterator[E]:
  for x in xs:
    if p(x): yield x

## Part 5: Infix operators
def 入x(cs, nam): return 入(eval(f'lambda x: {cs}'), nam)
def op_postfix(op, nam=None): return 入x(f'x {op}', nam or f'{(op)}')
def op_prefix(op, nam=None): return 入x(f'{op}x', nam or f'{(op)}')
def op_infix(op, rhs=None, lhs=None) -> callable:
  simplenames: Iterator[str] = iter(chrange('x', 'z'))
  anames = zip(simplenames, [lhs, rhs]); anamed: Dict[str, Any] = dict(anames)
  argt: Iterable[Tuple[(str, Any)]] = filter(lambda x: compose(noneq, snd), entries(anamed))
  tabl: List[str] = list(map(fst, argt)); atab = ','.join(tabl)
  body = f'{anamed["x"] or "x"}{op}{anamed["y"] or "y"}'
  desc = 'λ'+' '.join(tabl)+'. '+body if len(tabl) !=0 else body
  return 入(eval(f'lambda {atab}: {body}'), desc)

## Part 6: Misc operators
fst: Callable[[Tuple[(E,E)]], E]
snd: Callable[[Tuple[(E,E)]], E]
fst, snd = (op_postfix('[0]', 'x₀'), op_postfix('[1]', 'x₁'))
def its(k): return 入(lambda o: o[k], f'[{k}]')
dentry = lambda d: lambda k: (k, d[k])
def entries(d: Dict[K, V]) -> Iterable[Tuple[(K, V)]]: return map(dentry(d), d.keys())
noneq = op_postfix('is None', 'none?')

def TODO(reason='TODO'): raise RuntimeError(reason)

from .ops import Iinfty
IInfinity = Iinfty()
def distrange(k, n=IInfinity, start=0): return range(start, k*n, k)
