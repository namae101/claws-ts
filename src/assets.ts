import * as fs from 'node:fs';
import * as path from 'node:path';

let cachedWordlist: string | null = null;
let cachedFreq: string | null = null;

export function getWordlistContent(): string {
  if (cachedWordlist) return cachedWordlist;

  if (typeof process !== 'undefined' && typeof process.cwd === 'function') {
    const candidates = [
      path.resolve(process.cwd(), 'example/data/wordlist.txt'),
      path.resolve(process.cwd(), 'example/public/data/wordlist.txt'),
      path.resolve(__dirname, 'data/wordlist.txt'),
      path.resolve(__dirname, 'public/data/wordlist.txt')
    ];
    for (const p of candidates) {
      if (fs.existsSync(p)) {
        cachedWordlist = fs.readFileSync(p, 'utf8');
        return cachedWordlist;
      }
    }
  }

  return '';
}

export function getFrequencyContent(): string {
  if (cachedFreq) return cachedFreq;

  if (typeof process !== 'undefined' && typeof process.cwd === 'function') {
    const candidates = [
      path.resolve(process.cwd(), 'example/data/khmer_words_freq.tsv'),
      path.resolve(process.cwd(), 'example/public/data/khmer_words_freq.tsv'),
      path.resolve(__dirname, 'data/khmer_words_freq.tsv'),
      path.resolve(__dirname, 'public/data/khmer_words_freq.tsv')
    ];
    for (const p of candidates) {
      if (fs.existsSync(p)) {
        cachedFreq = fs.readFileSync(p, 'utf8');
        return cachedFreq;
      }
    }
  }

  return '';
}

export async function fetchWordlistAsync(url: string = './data/wordlist.txt'): Promise<string> {
  if (cachedWordlist) return cachedWordlist;
  const res = await fetch(url);
  cachedWordlist = await res.text();
  return cachedWordlist;
}

export async function fetchFrequencyAsync(url: string = './data/khmer_words_freq.tsv'): Promise<string> {
  if (cachedFreq) return cachedFreq;
  const res = await fetch(url);
  cachedFreq = await res.text();
  return cachedFreq;
}
