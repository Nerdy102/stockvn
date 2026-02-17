from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.indicators import ema, rsi, sma


Token = Tuple[str, str]

_TOKEN_RE = re.compile(
    r"""
    (?P<SPACE>\s+)
  | (?P<NUMBER>\d+(?:\.\d+)?)
  | (?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<OP>==|!=|>=|<=|[()*,/+\-<>])
    """,
    re.VERBOSE,
)


def tokenize(expr: str) -> List[Token]:
    tokens: List[Token] = []
    pos = 0
    while pos < len(expr):
        m = _TOKEN_RE.match(expr, pos)
        if not m:
            raise ValueError(f"Unexpected char at pos {pos}: {expr[pos:pos+20]!r}")
        kind = m.lastgroup or ""
        text = m.group(kind)
        pos = m.end()
        if kind == "SPACE":
            continue
        if kind == "IDENT":
            u = text.upper()
            if u in {"AND", "OR", "NOT"}:
                tokens.append(("LOGIC", u))
            else:
                tokens.append(("IDENT", text))
            continue
        if kind == "NUMBER":
            tokens.append(("NUMBER", text))
            continue
        tokens.append(("OP", text))
    tokens.append(("EOF", ""))
    return tokens


class Node:
    def eval(self, df: pd.DataFrame, cache: Dict[str, pd.Series]) -> pd.Series:
        raise NotImplementedError


@dataclass
class Number(Node):
    value: float

    def eval(self, df: pd.DataFrame, cache: Dict[str, pd.Series]) -> pd.Series:
        return pd.Series(self.value, index=df.index, dtype=float)


@dataclass
class SeriesRef(Node):
    name: str

    def eval(self, df: pd.DataFrame, cache: Dict[str, pd.Series]) -> pd.Series:
        if self.name not in df.columns:
            raise KeyError(f"Unknown series: {self.name}")
        return pd.to_numeric(df[self.name], errors="coerce").astype(float)


@dataclass
class Unary(Node):
    op: str
    rhs: Node

    def eval(self, df: pd.DataFrame, cache: Dict[str, pd.Series]) -> pd.Series:
        v = self.rhs.eval(df, cache)
        if self.op == "-":
            return -v
        if self.op == "NOT":
            return (~v.astype(bool)).astype(bool)
        raise ValueError(f"Unsupported unary op: {self.op}")


@dataclass
class Binary(Node):
    op: str
    lhs: Node
    rhs: Node

    def eval(self, df: pd.DataFrame, cache: Dict[str, pd.Series]) -> pd.Series:
        a = self.lhs.eval(df, cache)
        b = self.rhs.eval(df, cache)

        if self.op == "+":
            return a + b
        if self.op == "-":
            return a - b
        if self.op == "*":
            return a * b
        if self.op == "/":
            return a / b.replace(0, np.nan)

        if self.op == ">":
            return (a > b).astype(bool)
        if self.op == "<":
            return (a < b).astype(bool)
        if self.op == ">=":
            return (a >= b).astype(bool)
        if self.op == "<=":
            return (a <= b).astype(bool)
        if self.op == "==":
            return (a == b).astype(bool)
        if self.op == "!=":
            return (a != b).astype(bool)

        if self.op == "AND":
            return (a.astype(bool) & b.astype(bool)).astype(bool)
        if self.op == "OR":
            return (a.astype(bool) | b.astype(bool)).astype(bool)

        raise ValueError(f"Unsupported binary op: {self.op}")


@dataclass
class Func(Node):
    name: str
    args: List[Node]

    def eval(self, df: pd.DataFrame, cache: Dict[str, pd.Series]) -> pd.Series:
        fn = self.name.upper()
        if fn == "SMA":
            return _fn_ma(df, cache, kind="sma", args=self.args)
        if fn == "EMA":
            return _fn_ma(df, cache, kind="ema", args=self.args)
        if fn == "RSI":
            return _fn_rsi(df, cache, args=self.args)
        if fn == "AVG":
            return _fn_avg(df, cache, args=self.args)
        if fn == "REF":
            return _fn_ref(df, cache, args=self.args)
        if fn == "CROSSOVER":
            return _fn_crossover(df, cache, args=self.args)
        if fn == "CROSSUNDER":
            return _fn_crossunder(df, cache, args=self.args)
        raise ValueError(f"Unknown function: {self.name}")


def _as_int(node: Node, df: pd.DataFrame, cache: Dict[str, pd.Series]) -> int:
    s = node.eval(df, cache)
    v = float(s.iloc[-1])
    return int(round(v))


def _fn_ma(df: pd.DataFrame, cache: Dict[str, pd.Series], kind: str, args: List[Node]) -> pd.Series:
    # SMA(20) => SMA(close, 20)
    if len(args) == 1:
        window = _as_int(args[0], df, cache)
        src = pd.to_numeric(df["close"], errors="coerce").astype(float)
    elif len(args) == 2:
        src = args[0].eval(df, cache).astype(float)
        window = _as_int(args[1], df, cache)
    else:
        raise ValueError("SMA/EMA expects 1 or 2 arguments")
    return sma(src, window) if kind == "sma" else ema(src, window)


def _fn_rsi(df: pd.DataFrame, cache: Dict[str, pd.Series], args: List[Node]) -> pd.Series:
    if len(args) == 1:
        window = _as_int(args[0], df, cache)
        src = pd.to_numeric(df["close"], errors="coerce").astype(float)
    elif len(args) == 2:
        src = args[0].eval(df, cache).astype(float)
        window = _as_int(args[1], df, cache)
    else:
        raise ValueError("RSI expects 1 or 2 arguments")
    return rsi(src, window)


def _fn_avg(df: pd.DataFrame, cache: Dict[str, pd.Series], args: List[Node]) -> pd.Series:
    if len(args) != 2:
        raise ValueError("AVG expects 2 arguments")
    src = args[0].eval(df, cache).astype(float)
    window = _as_int(args[1], df, cache)
    return src.rolling(window=window, min_periods=window).mean()


def _fn_ref(df: pd.DataFrame, cache: Dict[str, pd.Series], args: List[Node]) -> pd.Series:
    if len(args) != 2:
        raise ValueError("REF expects 2 arguments")
    src = args[0].eval(df, cache).astype(float)
    n = _as_int(args[1], df, cache)
    return src.shift(n)


def _fn_crossover(df: pd.DataFrame, cache: Dict[str, pd.Series], args: List[Node]) -> pd.Series:
    if len(args) != 2:
        raise ValueError("CROSSOVER expects 2 arguments")
    a = args[0].eval(df, cache).astype(float)
    b = args[1].eval(df, cache).astype(float)
    out = (a.shift(1) <= b.shift(1)) & (a > b)
    return out.fillna(False).astype(bool)


def _fn_crossunder(df: pd.DataFrame, cache: Dict[str, pd.Series], args: List[Node]) -> pd.Series:
    if len(args) != 2:
        raise ValueError("CROSSUNDER expects 2 arguments")
    a = args[0].eval(df, cache).astype(float)
    b = args[1].eval(df, cache).astype(float)
    out = (a.shift(1) >= b.shift(1)) & (a < b)
    return out.fillna(False).astype(bool)


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.i = 0

    def _peek(self) -> Token:
        return self.tokens[self.i]

    def _eat(self, ttype: str, value: Optional[str] = None) -> Token:
        tok = self._peek()
        if tok[0] != ttype:
            raise ValueError(f"Expected {ttype}, got {tok}")
        if value is not None and tok[1] != value:
            raise ValueError(f"Expected {value}, got {tok}")
        self.i += 1
        return tok

    def parse(self) -> Node:
        node = self._parse_or()
        self._eat("EOF")
        return node

    def _parse_or(self) -> Node:
        node = self._parse_and()
        while self._peek() == ("LOGIC", "OR"):
            self._eat("LOGIC", "OR")
            node = Binary("OR", node, self._parse_and())
        return node

    def _parse_and(self) -> Node:
        node = self._parse_compare()
        while self._peek() == ("LOGIC", "AND"):
            self._eat("LOGIC", "AND")
            node = Binary("AND", node, self._parse_compare())
        return node

    def _parse_compare(self) -> Node:
        node = self._parse_add()
        while self._peek()[0] == "OP" and self._peek()[1] in {">", "<", ">=", "<=", "==", "!="}:
            op = self._eat("OP")[1]
            node = Binary(op, node, self._parse_add())
        return node

    def _parse_add(self) -> Node:
        node = self._parse_mul()
        while self._peek()[0] == "OP" and self._peek()[1] in {"+", "-"}:
            op = self._eat("OP")[1]
            node = Binary(op, node, self._parse_mul())
        return node

    def _parse_mul(self) -> Node:
        node = self._parse_unary()
        while self._peek()[0] == "OP" and self._peek()[1] in {"*", "/"}:
            op = self._eat("OP")[1]
            node = Binary(op, node, self._parse_unary())
        return node

    def _parse_unary(self) -> Node:
        tok = self._peek()
        if tok == ("LOGIC", "NOT"):
            self._eat("LOGIC", "NOT")
            return Unary("NOT", self._parse_unary())
        if tok == ("OP", "-"):
            self._eat("OP", "-")
            return Unary("-", self._parse_unary())
        return self._parse_primary()

    def _parse_primary(self) -> Node:
        tok = self._peek()
        if tok[0] == "NUMBER":
            self._eat("NUMBER")
            return Number(float(tok[1]))
        if tok == ("OP", "("):
            self._eat("OP", "(")
            node = self._parse_or()
            self._eat("OP", ")")
            return node
        if tok[0] == "IDENT":
            name = self._eat("IDENT")[1]
            if self._peek() == ("OP", "("):
                self._eat("OP", "(")
                args: List[Node] = []
                if self._peek() != ("OP", ")"):
                    args.append(self._parse_or())
                    while self._peek() == ("OP", ","):
                        self._eat("OP", ",")
                        args.append(self._parse_or())
                self._eat("OP", ")")
                return Func(name, args)
            return SeriesRef(name)
        raise ValueError(f"Unexpected token: {tok}")


def parse(expr: str) -> Node:
    return Parser(tokenize(expr)).parse()


def evaluate(expr: str, df: pd.DataFrame) -> pd.Series:
    """Evaluate DSL expression on OHLCV DataFrame."""
    node = parse(expr)
    cache: Dict[str, pd.Series] = {}
    return node.eval(df, cache)
