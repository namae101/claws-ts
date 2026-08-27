import { describe, it, expect } from 'vitest';
import {
  segKcc,
  isKhmerChar,
  isStartOfKcc,
  KHCONST,
  KHVOWEL,
  KHSUB,
  KHDIAC,
  KHSYM,
  KHNUMBER,
  KHLUNAR,
  EN_CHARS
} from '../src/kccs';

describe('kccs', () => {
  describe('character sets and helpers', () => {
    it('identifies Khmer characters correctly', () => {
      expect(isKhmerChar('ក')).toBe(true);
      expect(isKhmerChar('្')).toBe(true);
      expect(isKhmerChar('ា')).toBe(true);
      expect(isKhmerChar('៕')).toBe(true);
      expect(isKhmerChar('᧠')).toBe(true);
      expect(isKhmerChar('a')).toBe(false);
      expect(isKhmerChar('1')).toBe(false);
    });

    it('identifies start of KCC correctly', () => {
      expect(isStartOfKcc('ក')).toBe(true);
      expect(isStartOfKcc('ា')).toBe(false); // vowel is not start of KCC
      expect(isStartOfKcc('្')).toBe(false); // sub sign is not start of KCC
      expect(isStartOfKcc('a')).toBe(true);  // non-Khmer is considered start
    });
  });

  describe('segKcc', () => {
    it('segments standard Khmer phrases into correct KCCs', () => {
      const text = 'ខ្ញុំស្រឡាញ់ប្រទេសកម្ពុជា';
      const kccs = segKcc(text);
      expect(kccs).toEqual([
        'ខ្ញុំ', 'ស្រ', 'ឡា', 'ញ់', 'ប្រ', 'ទេ', 'ស', 'ក', 'ម្ពុ', 'ជា'
      ]);
    });

    it('handles subscript consonants properly', () => {
      const text = 'សង្គ្រាម';
      const kccs = segKcc(text);
      expect(kccs).toEqual(['ស', 'ង្គ្រា', 'ម']);
    });

    it('clusters numbers together', () => {
      expect(segKcc('១២៣៤៥')).toEqual(['១២៣៤៥']);
      expect(segKcc('12345')).toEqual(['12345']);
    });

    it('clusters non-Khmer English words together', () => {
      expect(segKcc('Hello world')).toEqual(['Hello', ' ', 'world']);
    });

    it('handles mixed English, numbers, and Khmer', () => {
      const text = 'Hello ខ្ញុំ 123 world!';
      expect(segKcc(text)).toEqual(['Hello', ' ', 'ខ្ញុំ', ' ', '123', ' ', 'world!']);
    });

    it('handles punctuation and symbols', () => {
      const text = 'សួស្តី! តើអ្នកសុខសប្បាយទេ?';
      const kccs = segKcc(text);
      expect(kccs).toEqual([
        'សួ', 'ស្តី', '!', ' ',
        'តើ', 'អ្ន', 'ក', 'សុ',
        'ខ', 'ស', 'ប្បា', 'យ',
        'ទេ', '?'
      ]);
    });

    it('handles empty string and whitespace', () => {
      expect(segKcc('')).toEqual([]);
      expect(segKcc('   ')).toEqual([' ', ' ', ' ']);
    });

    it('handles zero-width spaces', () => {
      const text = 'ខ្ញុំ\u200bស្រឡាញ់';
      const kccs = segKcc(text);
      expect(kccs).toEqual(['ខ្ញុំ', 'ស្រ', 'ឡា', 'ញ់']);
    });
  });
});
