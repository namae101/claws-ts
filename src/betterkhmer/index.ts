// Copyright (c) 2021-2024, SIL Global. Licensed under MIT license.
// Ported to TypeScript from betterkhmer. Regex-free.

export enum Cats {
  Other = 0,
  Base = 1,
  Robat = 2,
  Coeng = 3,
  Shift = 4,
  Z = 5,
  VPre = 6,
  VB = 7,
  VA = 8,
  VPost = 9,
  MS = 10,
  MF = 11,
  ZFCoeng = 12,
}

const CATEGORIES: Cats[] = [
  ...Array<Cats>(35).fill(Cats.Base), // 1780-17A2
  ...Array<Cats>(2).fill(Cats.Other), // 17A3-17A4
  ...Array<Cats>(15).fill(Cats.Base), // 17A5-17B3
  ...Array<Cats>(2).fill(Cats.Other), // 17B4-17B5
  Cats.VPost, // 17B6
  ...Array<Cats>(4).fill(Cats.VA), // 17B7-17BA
  ...Array<Cats>(3).fill(Cats.VB), // 17BB-17BD
  ...Array<Cats>(8).fill(Cats.VPre), // 17BE-17C5
  Cats.MS, // 17C6
  ...Array<Cats>(2).fill(Cats.MF), // 17C7-17C8
  ...Array<Cats>(2).fill(Cats.Shift), // 17C9-17CA
  Cats.MS, // 17CB
  Cats.Robat, // 17CC
  ...Array<Cats>(5).fill(Cats.MS), // 17CD-17D1
  Cats.Coeng, // 17D2
  Cats.MS, // 17D3
  ...Array<Cats>(9).fill(Cats.Other), // 17D4-17DC
  Cats.MS, // 17DD
];

const ZWNJ = 0x200c;
const ZWJ = 0x200d;
const COENG = 0x17d2;
const ROBAT = 0x17cc;
const BA = 0x1794;

function charcat(o: number): Cats {
  if (o >= 0x1780 && o <= 0x17dd) {
    return CATEGORIES[o - 0x1780]!;
  }
  if (o === ZWNJ) return Cats.Z;
  if (o === ZWJ) return Cats.ZFCoeng;
  return Cats.Other;
}

function isBase(r: number): boolean {
  return (
    (r >= 0x1780 && r <= 0x17a2) ||
    (r >= 0x17a5 && r <= 0x17b3) ||
    r === 0x25cc
  );
}

function isNonRo(r: number): boolean {
  return (
    (r >= 0x1780 && r <= 0x1799) ||
    (r >= 0x179b && r <= 0x17a2) ||
    (r >= 0x17a5 && r <= 0x17b3)
  );
}

function isNonBa(r: number): boolean {
  return (
    (r >= 0x1780 && r <= 0x1793) ||
    (r >= 0x1795 && r <= 0x17a2) ||
    (r >= 0x17a5 && r <= 0x17b3)
  );
}

function isS1(r: number): boolean {
  return (
    (r >= 0x1780 && r <= 0x1783) ||
    (r >= 0x1785 && r <= 0x1788) ||
    (r >= 0x178a && r <= 0x178d) ||
    (r >= 0x178f && r <= 0x1792) ||
    (r >= 0x1795 && r <= 0x1797) ||
    (r >= 0x179e && r <= 0x17a0) ||
    r === 0x17a2
  );
}

function isS2(r: number): boolean {
  return (
    r === 0x1780 ||
    r === 0x1784 ||
    r === 0x178e ||
    r === 0x1793 ||
    r === 0x1794 ||
    r === 0x17a1 ||
    (r >= 0x1798 && r <= 0x179d) ||
    (r >= 0x17a3 && r <= 0x17b3)
  );
}

function isVPre(r: number): boolean {
  return r >= 0x17c1 && r <= 0x17c5;
}

function isDigit(r: number): boolean {
  return r >= 0x17e0 && r <= 0x17e9;
}

function optRobat(r: number[], p: number): number[] {
  if (p < r.length && r[p] === ROBAT) {
    return [p, p + 1];
  }
  return [p];
}

function coengEnds(r: number[], s: number): number[] {
  const n = r.length;
  const res: number[] = [];
  if (s + 1 < n && r[s] === COENG && isBase(r[s + 1]!)) {
    res.push(s + 2);
  }
  if (
    s + 3 < n &&
    r[s] === COENG &&
    isNonRo(r[s + 1]!) &&
    r[s + 2] === COENG &&
    isBase(r[s + 3]!)
  ) {
    res.push(s + 4);
  }
  return res;
}

function strongEnds(r: number[], s: number, add: (e: number) => void): void {
  const n = r.length;
  if (s >= n) return;
  if (isS1(r[s]!)) {
    for (const p of optRobat(r, s + 1)) {
      add(p);
      if (p + 1 < n && r[p] === COENG && isNonBa(r[p + 1]!)) {
        const q = p + 2;
        add(q);
        if (q + 1 < n && r[q] === COENG && isNonBa(r[q + 1]!)) {
          add(q + 2);
        }
      }
    }
  }
  if (isNonBa(r[s]!)) {
    for (const p of optRobat(r, s + 1)) {
      if (p + 1 < n && r[p] === COENG && isS1(r[p + 1]!)) {
        const q = p + 2;
        add(q);
        if (q + 1 < n && r[q] === COENG && isNonBa(r[q + 1]!)) {
          add(q + 2);
        }
      }
      if (
        p + 3 < n &&
        r[p] === COENG &&
        isNonBa(r[p + 1]!) &&
        r[p + 2] === COENG &&
        isS1(r[p + 3]!)
      ) {
        add(p + 4);
      }
    }
  }
}

function nstrongEnds(r: number[], s: number, add: (e: number) => void): void {
  const n = r.length;
  if (s >= n) return;
  if (isS2(r[s]!)) {
    for (const p of optRobat(r, s + 1)) {
      add(p);
      if (p + 1 < n && r[p] === COENG && isS2(r[p + 1]!)) {
        const q = p + 2;
        add(q);
        if (q + 1 < n && r[q] === COENG && isS2(r[q + 1]!)) {
          add(q + 2);
        }
      }
    }
  }
  if (r[s] === BA) {
    for (const p of optRobat(r, s + 1)) {
      add(p);
      for (const e1 of coengEnds(r, p)) {
        add(e1);
        for (const e2 of coengEnds(r, e1)) {
          add(e2);
        }
      }
    }
  }
  if (isBase(r[s]!)) {
    for (const p of optRobat(r, s + 1)) {
      if (
        p + 3 < n &&
        r[p] === COENG &&
        isNonRo(r[p + 1]!) &&
        r[p + 2] === COENG &&
        r[p + 3] === BA
      ) {
        add(p + 4);
      }
      if (
        p + 3 < n &&
        r[p] === COENG &&
        r[p + 1] === BA &&
        r[p + 2] === COENG &&
        isBase(r[p + 3]!)
      ) {
        add(p + 4);
      }
    }
  }
}

function canEndAt(
  r: number[],
  target: number,
  ends: (r: number[], s: number, add: (e: number) => void) => void
): boolean {
  for (let s = 0; s < target; s++) {
    let found = false;
    ends(r, s, (e) => {
      if (e === target) found = true;
    });
    if (found) return true;
  }
  return false;
}

function vaSamyokAt(r: number[], p: number): boolean {
  const n = r.length;
  if (p >= n) return false;
  const c = r[p]!;
  if (c === 0x17d0) return true;
  if (c >= 0x17b7 && c <= 0x17ba) return true;
  if (c === 0x17be || c === 0x17bf || c === 0x17dd) return true;
  if (c === 0x17b6 && p + 1 < n && r[p + 1] === 0x17c6) return true;
  return false;
}

function applyShifter(
  r: number[],
  ends: (r: number[], s: number, add: (e: number) => void) => void,
  shifter: number
): void {
  for (let k = 0; k < r.length; k++) {
    if (r[k] !== 0x17bb) continue;
    const ctx =
      canEndAt(r, k, ends) ||
      (k >= 1 && isVPre(r[k - 1]!) && canEndAt(r, k - 1, ends));
    if (ctx && vaSamyokAt(r, k + 1)) {
      r[k] = shifter;
    }
  }
}

function collapseInvis(r: number[]): number[] {
  const n = r.length;
  const out: number[] = [];
  let i = 0;
  while (i < n) {
    let g1End = -1;
    if (r[i] === ZWJ && i + 1 < n && r[i + 1] === COENG) {
      g1End = i + 2;
    } else if (r[i] === COENG) {
      g1End = i + 1;
    }
    if (g1End >= 0) {
      let k = g1End;
      while (k < n && (r[k] === COENG || r[k] === ZWNJ || r[k] === ZWJ)) {
        k++;
      }
      if (k > g1End) {
        for (let idx = i; idx < g1End; idx++) out.push(r[idx]!);
        i = k;
        continue;
      }
    }
    out.push(r[i]!);
    i++;
  }
  return out;
}

function pairReplace(r: number[], a: number, b: number, ...repl: number[]): number[] {
  const n = r.length;
  const out: number[] = [];
  let i = 0;
  while (i < n) {
    if (i + 1 < n && r[i] === a && r[i + 1] === b) {
      out.push(...repl);
      i += 2;
      continue;
    }
    out.push(r[i]!);
    i++;
  }
  return out;
}

function pairReplace3(
  r: number[],
  a: number,
  b: number,
  c: number,
  repl: number
): number[] {
  const n = r.length;
  const out: number[] = [];
  let i = 0;
  while (i < n) {
    if (i + 2 < n && r[i] === a && r[i + 1] === b && r[i + 2] === c) {
      out.push(repl);
      i += 3;
      continue;
    }
    out.push(r[i]!);
    i++;
  }
  return out;
}

function vowelSplit(r: number[], tail: number, head: number): number[] {
  const n = r.length;
  const out: number[] = [];
  let i = 0;
  while (i < n) {
    if (r[i] === 0x17c1) {
      if (i + 2 < n && r[i + 1]! >= 0x17bb && r[i + 1]! <= 0x17bd && r[i + 2] === tail) {
        out.push(head, r[i + 1]!);
        i += 3;
        continue;
      }
      if (i + 1 < n && r[i + 1] === tail) {
        out.push(head);
        i += 2;
        continue;
      }
    }
    out.push(r[i]!);
    i++;
  }
  return out;
}

function coengRo(r: number[]): number[] {
  const n = r.length;
  const out: number[] = [];
  let i = 0;
  while (i < n) {
    if (
      i + 3 < n &&
      r[i] === COENG &&
      r[i + 1] === 0x179a &&
      r[i + 2] === COENG &&
      r[i + 3]! >= 0x1780 &&
      r[i + 3]! <= 0x17b3
    ) {
      out.push(r[i + 2]!, r[i + 3]!, r[i]!, r[i + 1]!);
      i += 4;
      continue;
    }
    out.push(r[i]!);
    i++;
  }
  return out;
}

function coengDa(r: number[]): number[] {
  const n = r.length;
  const out: number[] = [];
  let i = 0;
  while (i < n) {
    if (i + 1 < n && r[i] === COENG && r[i + 1] === 0x178a) {
      out.push(COENG, 0x178f);
      i += 2;
      continue;
    }
    out.push(r[i]!);
    i++;
  }
  return out;
}

function lunar1(r: number[]): number[] {
  const n = r.length;
  const out: number[] = [];
  let i = 0;
  while (i < n) {
    if (
      r[i] === 0x17e1 &&
      i + 3 < n &&
      isDigit(r[i + 1]!) &&
      r[i + 2] === COENG &&
      r[i + 3] === 0x17d4
    ) {
      const v = 10 + (r[i + 1]! - 0x17e0);
      if (v > 15) {
        out.push(...r.slice(i, i + 4));
      } else {
        out.push(0x19e0 + v);
      }
      i += 4;
      continue;
    }
    if (
      i + 2 < n &&
      isDigit(r[i]!) &&
      r[i + 1] === COENG &&
      r[i + 2] === 0x17d4
    ) {
      out.push(0x19e0 + (r[i]! - 0x17e0));
      i += 3;
      continue;
    }
    out.push(r[i]!);
    i++;
  }
  return out;
}

function lunar2(r: number[]): number[] {
  const n = r.length;
  const out: number[] = [];
  let i = 0;
  while (i < n) {
    if (r[i] === 0x17d4 && i + 1 < n && r[i + 1] === COENG) {
      if (i + 3 < n && r[i + 2] === 0x17e1 && isDigit(r[i + 3]!)) {
        const v = 10 + (r[i + 3]! - 0x17e0);
        if (v > 15) {
          out.push(...r.slice(i, i + 4));
        } else {
          out.push(0x19f0 + v);
        }
        i += 4;
        continue;
      }
      if (i + 2 < n && isDigit(r[i + 2]!)) {
        out.push(0x19f0 + (r[i + 2]! - 0x17e0));
        i += 3;
        continue;
      }
    }
    out.push(r[i]!);
    i++;
  }
  return out;
}

function hasKhmer(cps: number[]): boolean {
  for (const o of cps) {
    if (o >= 0x1780 && o <= 0x17ff) return true;
  }
  return false;
}

/**
 * Return the Khmer-normalized form of txt.
 * - Fixes reordering of vowels, subscripts, shifters, diacritics, and lunar dates.
 * - Removes invisible zero-width spaces (ZWSP U+200B, ZWNJ U+200C, ZWJ U+200D, BOM U+FEFF).
 * - Replaces Khmer punctuation and symbols (U+17D4..U+17DB: ។, ៕, ៛, ៗ, ៚, ៙, ៘, ៖) with regular spaces.
 */
export function normalize(txt: string, lang: string = 'km'): string {
  if (!txt) return '';

  // 1. Remove invisible / zero-width characters (ZWSP, ZWNJ, ZWJ, BOM)
  let cleaned = txt.replace(/[\u200B\u200C\u200D\uFEFF]/g, '');

  // 2. Replace Khmer punctuation and symbols (U+17D4..U+17DB) with regular space
  cleaned = cleaned.replace(/[\u17D4-\u17DB]/g, ' ');

  // Normalize contiguous spaces into single space
  cleaned = cleaned.replace(/ +/g, ' ');

  let cps: number[] = [];
  for (let i = 0; i < cleaned.length; i++) {
    cps.push(cleaned.charCodeAt(i));
  }
  if (lang === 'xhm') {
    const out: number[] = [];
    const n = cps.length;
    for (let i = 0; i < n; i++) {
      const o = cps[i]!;
      if (o >= 0x17b7 && o <= 0x17c5 && i + 1 < n && cps[i + 1] === COENG) {
        out.push(ZWJ);
      }
      out.push(o);
    }
    cps = out;
  }

  if (!hasKhmer(cps)) {
    return String.fromCharCode(...cps);
  }

  const n = cps.length;
  const cats: Cats[] = cps.map((o) => charcat(o));

  for (let i = 1; i < n; i++) {
    if (cps[i - 1] === ZWJ || cps[i - 1] === COENG) {
      if (cats[i] === Cats.Base || cats[i] === Cats.Coeng) {
        cats[i] = cats[i - 1]!;
      }
    }
  }

  let i = 0;
  const res: number[] = [];
  while (i < n) {
    if (cats[i] !== Cats.Base) {
      res.push(cps[i]!);
      i++;
      continue;
    }
    let j = i + 1;
    while (j < n && cats[j]! > Cats.Base) {
      j++;
    }

    const indices = Array.from({ length: j - i }, (_, idx) => i + idx).sort(
      (a, b) => {
        const catA = cats[a]!;
        const catB = cats[b]!;
        if (catA !== catB) return catA - catB;
        return a - b;
      }
    );

    let syl = indices.map((k) => cps[k]!);

    syl = collapseInvis(syl);
    syl = pairReplace(syl, 0x17be, 0x17b6, 0x17c4, 0x17b8); // ើា -> ោី
    syl = vowelSplit(syl, 0x17b8, 0x17be); // េ(◌)ី -> ើ(◌)
    syl = vowelSplit(syl, 0x17b6, 0x17c4); // េ(◌)ា -> ោ(◌)
    syl = pairReplace(syl, 0x17be, 0x17bb, 0x17bb, 0x17be); // ើុ -> ុើ
    applyShifter(syl, strongEnds, 0x17ca); // strong -u -> ៊
    applyShifter(syl, nstrongEnds, 0x17c9); // weak   -u -> ៉
    syl = coengRo(syl);
    syl = coengDa(syl);
    syl = lunar1(syl);
    syl = lunar2(syl);
    syl = pairReplace3(syl, 0x17d4, 0x17d2, 0x17d4, 0x19f0); // ។្។ -> ᧰

    res.push(...syl);
    i = j;
  }

  return String.fromCharCode(...res);
}
