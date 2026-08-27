export const KHCONST = new Set('កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវឝឞសហឡអឣឤឥឦឧឨឩឪឫឬឭឮឯឰឱឲឳ');
export const KHVOWEL = new Set('឴឵ាិីឹឺុូួើឿៀេែៃោៅ\u17c6\u17c7\u17c8');
export const KHSUB = new Set('្');
export const KHDIAC = new Set('\u17c9\u17ca\u17cb\u17cc\u17cd\u17ce\u17cf\u17d0');
export const KHSYM = new Set('៕។៛ៗ៚៙៘,.? ');
export const KHNUMBER = new Set('០១២៣៤៥៦៧៨៩0123456789');
export const KHLUNAR = new Set('᧠᧡᧢᧣᧤᧥᧦᧧᧨᧩᧪᧫᧬᧭᧮᧯᧰᧱᧲᧳᧴᧵᧶᧷᧸᧹᧺᧻᧼᧽᧾᧿');
export const EN_CHARS = new Set('abcdefghijklmnopqrstuvwxyz0123456789');

export function isKhmerChar(ch: string): boolean {
  if (ch >= '\u1780' && ch <= '\u17ff') return true;
  if (KHSYM.has(ch)) return true;
  if (KHLUNAR.has(ch)) return true;
  return false;
}

export function isStartOfKcc(ch: string): boolean {
  if (isKhmerChar(ch)) {
    if (KHCONST.has(ch)) return true;
    if (KHSYM.has(ch)) return true;
    if (KHNUMBER.has(ch)) return true;
    if (KHLUNAR.has(ch)) return true;
    return false;
  }
  return true;
}

export function segKcc(strSentence: string): string[] {
  const segs: string[] = [];
  let cur = '';

  const words = strSentence.split('\u200b');
  for (const word of words) {
    for (let i = 0; i < word.length; i++) {
      const c = word[i];
      cur += c;
      const nextchar = i + 1 < word.length ? word[i + 1] : '';

      // cluster non-khmer chars together
      if (!isKhmerChar(c) && nextchar !== ' ' && nextchar !== '' && !isKhmerChar(nextchar)) {
        continue;
      }
      // cluster numbers together
      if (KHNUMBER.has(c) && KHNUMBER.has(nextchar)) {
        continue;
      }

      // non-khmer character has no cluster
      if (!isKhmerChar(c) || nextchar === ' ' || nextchar === '') {
        segs.push(cur);
        cur = '';
      } else if (isStartOfKcc(nextchar) && !KHSUB.has(c)) {
        segs.push(cur);
        cur = '';
      }
    }
  }
  if (cur.length > 0) {
    segs.push(cur);
  }
  return segs;
}
