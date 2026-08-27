import {
  getFrequencyContent,
  getWordlistContent,
  fetchFrequencyAsync,
  fetchWordlistAsync,
} from '../assets.js';
import { normalize } from '../betterkhmer/index.js';
import {
  damerauLevenshteinDistance,
  isKhmerText,
  similarityRatio,
} from './distance.js';
import { PrefixTrie } from './trie.js';

export interface SuggestionItem {
  word: string;
  score: number;
  distance: number;
  frequency: number;
  match_type: 'prefix' | 'suffix' | 'spelling' | 'exact_match' | 'fuzzy';
}

export interface WordSpellCheckResult {
  word: string;
  normalized_word: string;
  is_correct: boolean;
  suggestions: string[];
}

export interface KhmerSpellCheckerOptions {
  dictionaryContent?: string;
  freqContent?: string;
  autoInit?: boolean;
}

export class KhmerSpellChecker {
  public words: Set<string> = new Set();
  public wordFreq: Record<string, number> = {};
  public deletes: Record<string, string[]> = {};
  public trie: PrefixTrie = new PrefixTrie();
  public suffixTrie: PrefixTrie = new PrefixTrie();
  public ngramIndex: Record<string, string[]> = {};
  private totalFreq: number = 1.0;
  private initPromise: Promise<void> | null = null;
  private dictionaryContent?: string;
  private freqContent?: string;

  constructor(options: KhmerSpellCheckerOptions = {}) {
    this.dictionaryContent = options.dictionaryContent;
    this.freqContent = options.freqContent;

    // Do NOT block module startup synchronously. Init on-demand or if requested.
    if (options.autoInit) {
      void this.init();
    }
  }

  public async init(): Promise<void> {
    if (this.words.size > 0) return;
    if (this.initPromise) return this.initPromise;

    this.initPromise = (async () => {
      let freqData = this.freqContent ?? getFrequencyContent();
      if (!freqData && typeof fetch !== 'undefined') {
        try {
          freqData = await fetchFrequencyAsync();
        } catch {
          // ignore
        }
      }
      if (freqData) {
        this.loadFrequenciesContent(freqData);
      }

      let dictData = this.dictionaryContent ?? getWordlistContent();
      if (!dictData && typeof fetch !== 'undefined') {
        try {
          dictData = await fetchWordlistAsync();
        } catch {
          // ignore
        }
      }
      if (dictData) {
        this.loadDictionaryContent(dictData);
      }
    })();

    await this.initPromise;
  }

  public loadFrequenciesContent(content: string): number {
    let count = 0;
    const lines = content.split('\n');

    for (let i = 1; i < lines.length; i++) {
      const line = lines[i]!.trim();
      if (!line) continue;
      const parts = line.split('\t');
      if (parts.length >= 2) {
        const w = normalize(parts[0]!.trim());
        if (w && isKhmerText(w)) {
          const freq = parseFloat(parts[1]!.trim());
          this.wordFreq[w] = Number.isNaN(freq) ? 1.0 : freq;
          count++;
        }
      }
    }
    this.recomputeTotalFreq();
    return count;
  }

  private recomputeTotalFreq(): void {
    let sum = 0;
    const keys = Object.keys(this.wordFreq);
    for (let i = 0; i < keys.length; i++) {
      sum += this.wordFreq[keys[i]!]!;
    }
    this.totalFreq = sum || 1.0;
  }

  public getDeletes(word: string, maxDistance: number = 1): Set<string> {
    const deletes = new Set<string>();
    let queue = [word];
    for (let d = 0; d < maxDistance; d++) {
      const tempQueue: string[] = [];
      for (const item of queue) {
        if (item.length > 1) {
          for (let i = 0; i < item.length; i++) {
            const del = item.slice(0, i) + item.slice(i + 1);
            if (!deletes.has(del)) {
              deletes.add(del);
              tempQueue.push(del);
            }
          }
        }
      }
      queue = tempQueue;
    }
    return deletes;
  }

  public getNgrams(s: string, n: number = 2): Set<string> {
    if (s.length < n) return new Set([s]);
    const ngrams = new Set<string>();
    for (let i = 0; i <= s.length - n; i++) {
      ngrams.add(s.slice(i, i + n));
    }
    return ngrams;
  }

  private indexWord(word: string, freq?: number): void {
    this.words.add(word);
    if (freq !== undefined) {
      this.wordFreq[word] = freq;
    } else if (this.wordFreq[word] === undefined) {
      this.wordFreq[word] = 1.0;
    }

    const f = this.wordFreq[word]!;
    this.trie.insert(word, f);

    const reversed = word.split('').reverse().join('');
    this.suffixTrie.insert(reversed, f);

    // Fast 1-deletion indexing (lightweight & fast)
    for (let i = 0; i < word.length; i++) {
      const d = word.slice(0, i) + word.slice(i + 1);
      if (!this.deletes[d]) {
        this.deletes[d] = [word];
      } else {
        this.deletes[d]!.push(word);
      }
    }

    // 2-gram indexing
    const ngrams = this.getNgrams(word, 2);
    for (const ng of ngrams) {
      if (!this.ngramIndex[ng]) {
        this.ngramIndex[ng] = [word];
      } else {
        this.ngramIndex[ng]!.push(word);
      }
    }
  }

  public loadDictionaryContent(content: string): number {
    let added = 0;
    const lines = content.split('\n');

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i]!.trim();
      if (!line || line.startsWith('#')) continue;
      const normWord = normalize(line);
      if (normWord && isKhmerText(normWord) && !this.words.has(normWord)) {
        this.indexWord(normWord);
        added++;
      }
    }
    this.recomputeTotalFreq();
    return added;
  }

  public addWords(words: string[]): number {
    let count = 0;
    for (const w of words) {
      const normWord = normalize(w.trim());
      if (normWord && isKhmerText(normWord) && !this.words.has(normWord)) {
        this.indexWord(normWord);
        count++;
      }
    }
    this.recomputeTotalFreq();
    return count;
  }

  public setWords(words: string[]): number {
    this.words.clear();
    this.wordFreq = {};
    this.deletes = {};
    this.trie.clear();
    this.suffixTrie.clear();
    this.ngramIndex = {};
    const count = this.addWords(words);
    this.recomputeTotalFreq();
    return count;
  }

  public getWordCount(): number {
    return this.words.size;
  }

  public isNonWord(token: string): boolean {
    if (!token || !token.trim()) return true;
    const cleaned = token.replace(
      /[\s\d\u17E0-\u17E9\u17D4-\u17DA.,!?:;\-_()[\]{}"'/\\«»“”]+/g,
      ''
    );
    return cleaned.length === 0;
  }

  // --- Trie-based Segmentation ---

  private getDag(text: string): Record<number, number[]> {
    const dag: Record<number, number[]> = {};
    const n = text.length;
    for (let i = 0; i < n; i++) {
      dag[i] = [];
      let curr = this.trie.root;
      let j = i;
      while (j < n && curr.children[text[j]!]) {
        curr = curr.children[text[j]!]!;
        if (curr.isEnd) {
          dag[i]!.push(j + 1);
        }
        j++;
      }
      if (dag[i]!.length === 0) {
        dag[i]!.push(i + 1);
      }
    }
    return dag;
  }

  private segmentKhmerViterbi(text: string): string[] {
    const n = text.length;
    if (n === 0) return [];

    const dag = this.getDag(text);
    const dp: Record<number, [score: number, nextIdx: number]> = {
      [n]: [0.0, 0],
    };
    const logTotal = Math.log(this.totalFreq || 1.0);
    const minLogProb = Math.log(0.1 / (this.totalFreq || 1.0));

    for (let i = n - 1; i >= 0; i--) {
      const edges = dag[i]!;
      let bestScore = -Infinity;
      let bestNext = i + 1;

      for (const j of edges) {
        const word = text.slice(i, j);
        const freq = this.wordFreq[word] ?? 0.0;
        const prob = freq > 0 ? Math.log(freq) - logTotal : minLogProb * (j - i);
        const totalScore = prob + dp[j]![0];
        if (totalScore > bestScore) {
          bestScore = totalScore;
          bestNext = j;
        }
      }
      dp[i] = [bestScore, bestNext];
    }

    const tokens: string[] = [];
    let i = 0;
    while (i < n) {
      const nextI = dp[i]![1];
      tokens.push(text.slice(i, nextI));
      i = nextI;
    }
    return tokens;
  }

  private segmentKhmerFmm(text: string): string[] {
    const tokens: string[] = [];
    const n = text.length;
    let i = 0;
    while (i < n) {
      let curr = this.trie.root;
      let longestEnd = -1;
      let j = i;
      while (j < n && curr.children[text[j]!]) {
        curr = curr.children[text[j]!]!;
        if (curr.isEnd) {
          longestEnd = j + 1;
        }
        j++;
      }

      if (longestEnd !== -1) {
        tokens.push(text.slice(i, longestEnd));
        i = longestEnd;
      } else {
        tokens.push(text[i]!);
        i++;
      }
    }
    return tokens;
  }

  public segment(text: string, algorithm: 'viterbi' | 'fmm' | string = 'viterbi'): string[] {
    if (!text) return [];

    const pattern = /([\u1780-\u17FF\u19E0-\u19FF]+|[^\u1780-\u17FF\u19E0-\u19FF\s]+|\s+)/g;
    const tokens: string[] = [];

    let match: RegExpExecArray | null;
    while ((match = pattern.exec(text)) !== null) {
      const chunk = match[0]!;
      if (!chunk || chunk.trim().length === 0) continue;

      if (isKhmerText(chunk)) {
        const segs =
          algorithm === 'fmm'
            ? this.segmentKhmerFmm(chunk)
            : this.segmentKhmerViterbi(chunk);
        tokens.push(...segs);
      } else {
        tokens.push(chunk);
      }
    }

    return tokens;
  }

  // --- Spell Checking & Suffix / Suggestion Search ---

  public getCandidates(normWord: string, maxDistance: number = 5): Set<string> {
    const candidates = new Set<string>();

    if (this.deletes[normWord]) {
      for (const w of this.deletes[normWord]!) candidates.add(w);
    }

    const deletes = this.getDeletes(normWord, 1);
    for (const d of deletes) {
      if (this.deletes[d]) {
        for (const w of this.deletes[d]!) candidates.add(w);
      }
    }

    if (candidates.size < 5) {
      const qGrams = this.getNgrams(normWord, 2);
      const counts: Record<string, number> = {};
      for (const g of qGrams) {
        if (this.ngramIndex[g]) {
          for (const w of this.ngramIndex[g]!) {
            counts[w] = (counts[w] ?? 0) + 1;
          }
        }
      }
      const minShared =
        qGrams.size <= 4 ? 1 : Math.max(1, Math.floor(qGrams.size / 3));
      const wordsWithGrams = Object.keys(counts);
      for (const w of wordsWithGrams) {
        if (
          counts[w]! >= minShared &&
          Math.abs(w.length - normWord.length) <= maxDistance
        ) {
          candidates.add(w);
        }
      }
    }

    return candidates;
  }

  public checkWord(
    word: string,
    maxSuggestions: number = 3,
    maxDistance: number = 5
  ): WordSpellCheckResult {
    const normWord = normalize(word.trim());

    if (!isKhmerText(normWord) || this.isNonWord(normWord)) {
      return {
        word,
        normalized_word: normWord,
        is_correct: true,
        suggestions: [],
      };
    }

    if (this.words.size === 0 || this.words.has(normWord)) {
      return {
        word,
        normalized_word: normWord,
        is_correct: true,
        suggestions: [],
      };
    }

    // Misspelled word: multi-strategy candidate gathering
    const suffixCands: Array<{
      dist: number;
      freq: number;
      score: number;
      len: number;
      word: string;
      type: 'suffix';
    }> = [];
    const prefixCands: Array<{
      dist: number;
      freq: number;
      score: number;
      len: number;
      word: string;
      type: 'prefix';
    }> = [];
    const spellCands: Array<{
      dist: number;
      freq: number;
      score: number;
      len: number;
      word: string;
      type: 'spelling';
    }> = [];
    const seen = new Set<string>();

    // Suffix candidates
    const reversed = normWord.split('').reverse().join('');
    const revMatches = this.suffixTrie.prefixSearch(
      reversed,
      maxSuggestions * 6
    );
    for (const [revC, freq] of revMatches) {
      const c = revC.split('').reverse().join('');
      if (c !== normWord) {
        const dist = damerauLevenshteinDistance(normWord, c);
        if (dist <= maxDistance) {
          const score = similarityRatio(normWord, c);
          suffixCands.push({
            dist,
            freq,
            score,
            len: c.length,
            word: c,
            type: 'suffix',
          });
        }
      }
    }
    suffixCands.sort((a, b) => {
      if (a.dist !== b.dist) return a.dist - b.dist;
      if (b.freq !== a.freq) return b.freq - a.freq;
      return b.score - a.score;
    });

    // Prefix candidates
    const prefixMatches = this.trie.prefixSearch(normWord, maxSuggestions * 6);
    for (const [c, freq] of prefixMatches) {
      if (c !== normWord) {
        const dist = damerauLevenshteinDistance(normWord, c);
        if (dist <= maxDistance) {
          const score = similarityRatio(normWord, c);
          prefixCands.push({
            dist,
            freq,
            score,
            len: c.length,
            word: c,
            type: 'prefix',
          });
        }
      }
    }
    prefixCands.sort((a, b) => {
      if (a.dist !== b.dist) return a.dist - b.dist;
      if (b.freq !== a.freq) return b.freq - a.freq;
      return b.score - a.score;
    });

    // Spelling candidates
    const deletesCands = this.getCandidates(normWord, maxDistance);
    for (const c of deletesCands) {
      if (c !== normWord) {
        const dist = damerauLevenshteinDistance(normWord, c);
        if (dist <= maxDistance) {
          const score = similarityRatio(normWord, c);
          const freq = this.wordFreq[c] ?? 1.0;
          spellCands.push({
            dist,
            freq,
            score,
            len: c.length,
            word: c,
            type: 'spelling',
          });
        }
      }
    }
    spellCands.sort((a, b) => {
      if (a.dist !== b.dist) return a.dist - b.dist;
      if (b.freq !== a.freq) return b.freq - a.freq;
      return b.score - a.score;
    });

    // Interleave sources
    const activeSources = [suffixCands, prefixCands, spellCands].filter(
      (s) => s.length > 0
    );
    const suggestions: string[] = [];
    let idx = 0;

    while (
      suggestions.length < maxSuggestions &&
      activeSources.some((s) => idx < s.length)
    ) {
      for (const s of activeSources) {
        if (idx < s.length) {
          const item = s[idx]!;
          if (!seen.has(item.word)) {
            seen.add(item.word);
            suggestions.push(item.word);
            if (suggestions.length >= maxSuggestions) break;
          }
        }
      }
      idx++;
    }

    return {
      word,
      normalized_word: normWord,
      is_correct: false,
      suggestions,
    };
  }

  public checkTokens(
    tokens: string[],
    maxSuggestions: number = 3,
    maxDistance: number = 5
  ): WordSpellCheckResult[] {
    return tokens.map((token) =>
      this.checkWord(token, maxSuggestions, maxDistance)
    );
  }

  public suggestSuffix(
    suffix: string,
    maxSuggestions: number = 5,
    minScore: number = 0.0,
    maxDistance: number = 5
  ): SuggestionItem[] {
    const normSuffix = normalize(suffix.trim());
    if (
      !normSuffix ||
      !isKhmerText(normSuffix) ||
      this.isNonWord(normSuffix)
    ) {
      return [];
    }

    const reversed = normSuffix.split('').reverse().join('');
    const revMatches = this.suffixTrie.prefixSearch(
      reversed,
      maxSuggestions * 8
    );
    const ranked: SuggestionItem[] = [];

    for (const [revW, freq] of revMatches) {
      const w = revW.split('').reverse().join('');
      const dist = damerauLevenshteinDistance(normSuffix, w);
      if (dist <= maxDistance) {
        const score = similarityRatio(normSuffix, w);
        if (score >= minScore) {
          ranked.push({
            word: w,
            score,
            distance: dist,
            frequency: freq,
            match_type: 'suffix',
          });
        }
      }
    }

    ranked.sort((a, b) => {
      if (a.distance !== b.distance) return a.distance - b.distance;
      if (b.frequency !== a.frequency) return b.frequency - a.frequency;
      return b.score - a.score;
    });

    return ranked.slice(0, maxSuggestions);
  }

  public suggestPrefix(
    prefix: string,
    maxSuggestions: number = 5,
    minScore: number = 0.0,
    maxDistance: number = 5
  ): SuggestionItem[] {
    return this.suggest(prefix, maxSuggestions, maxDistance, minScore, 'prefix');
  }

  public suggestSpelling(
    query: string,
    maxSuggestions: number = 5,
    maxDistance: number = 5,
    minScore: number = 0.0
  ): SuggestionItem[] {
    return this.suggest(query, maxSuggestions, maxDistance, minScore, 'spelling');
  }
  public suggest(
    query: string,
    maxSuggestions: number = 5,
    maxDistance: number = 5,
    minScore: number = 0.0,
    mode: 'auto' | 'prefix' | 'suffix' | 'spelling' | 'fuzzy' | string = 'auto'
  ): SuggestionItem[] {
    const normQuery = normalize(query.trim());
    if (!normQuery || !isKhmerText(normQuery) || this.isNonWord(normQuery)) {
      return [];
    }

    // 1. Suffix mode
    if (mode === 'suffix') {
      return this.suggestSuffix(
        normQuery,
        maxSuggestions,
        minScore,
        maxDistance
      );
    }

    // 2. Prefix mode
    if (mode === 'prefix') {
      const prefixMatches = this.trie.prefixSearch(
        normQuery,
        maxSuggestions * 2
      );
      const results: SuggestionItem[] = [];
      for (const [w, freq] of prefixMatches) {
        const dist = damerauLevenshteinDistance(normQuery, w);
        const score = similarityRatio(normQuery, w);
        if (score >= minScore) {
          results.push({
            word: w,
            score,
            distance: dist,
            frequency: freq,
            match_type: dist === 0 ? 'exact_match' : 'prefix',
          });
          if (results.length >= maxSuggestions) break;
        }
      }
      return results;
    }

    // 3. Spelling mode
    if (mode === 'spelling') {
      const cands = this.getCandidates(normQuery, maxDistance);
      const spellingResults: SuggestionItem[] = [];
      for (const cand of cands) {
        const dist = damerauLevenshteinDistance(normQuery, cand);
        if (dist <= maxDistance) {
          const score = similarityRatio(normQuery, cand);
          if (score >= minScore) {
            const freq = this.wordFreq[cand] ?? 1.0;
            spellingResults.push({
              word: cand,
              score,
              distance: dist,
              frequency: freq,
              match_type: 'spelling',
            });
          }
        }
      }
      spellingResults.sort((a, b) => {
        if (a.distance !== b.distance) return a.distance - b.distance;
        if (b.frequency !== a.frequency) return b.frequency - a.frequency;
        return b.score - a.score;
      });
      return spellingResults.slice(0, maxSuggestions);
    }

    // 4. Fuzzy mode
    if (mode === 'fuzzy') {
      const qGrams = this.getNgrams(normQuery, 2);
      const candidateCounts: Record<string, number> = {};
      for (const g of qGrams) {
        if (this.ngramIndex[g]) {
          for (const w of this.ngramIndex[g]!) {
            candidateCounts[w] = (candidateCounts[w] ?? 0) + 1;
          }
        }
      }
      const minShared =
        qGrams.size <= 4 ? 1 : Math.max(1, Math.floor(qGrams.size / 3));
      const words = Object.keys(candidateCounts);
      const ranked: SuggestionItem[] = [];
      for (const cand of words) {
        if (candidateCounts[cand]! >= minShared) {
          const dist = damerauLevenshteinDistance(normQuery, cand);
          if (dist <= maxDistance) {
            const score = similarityRatio(normQuery, cand);
            if (score >= minScore) {
              const freq = this.wordFreq[cand] ?? 1.0;
              ranked.push({
                word: cand,
                score,
                distance: dist,
                frequency: freq,
                match_type: 'fuzzy',
              });
            }
          }
        }
      }
      ranked.sort((a, b) => {
        if (a.distance !== b.distance) return a.distance - b.distance;
        if (b.frequency !== a.frequency) return b.frequency - a.frequency;
        return b.score - a.score;
      });
      return ranked.slice(0, maxSuggestions);
    }

    // 5. Auto mode (interleaving)
    const suffixCands: SuggestionItem[] = [];
    const prefixCands: SuggestionItem[] = [];
    const spellCands: SuggestionItem[] = [];
    const seen = new Set<string>();

    // (a) Suffix matches
    const reversed = normQuery.split('').reverse().join('');
    const revMatches = this.suffixTrie.prefixSearch(
      reversed,
      maxSuggestions * 6
    );
    for (const [revC, freq] of revMatches) {
      const c = revC.split('').reverse().join('');
      const dist = damerauLevenshteinDistance(normQuery, c);
      if (dist <= maxDistance) {
        const score = similarityRatio(normQuery, c);
        if (score >= minScore) {
          suffixCands.push({
            word: c,
            score,
            distance: dist,
            frequency: freq,
            match_type: 'suffix',
          });
        }
      }
    }
    suffixCands.sort((a, b) => {
      if (a.distance !== b.distance) return a.distance - b.distance;
      if (b.frequency !== a.frequency) return b.frequency - a.frequency;
      return b.score - a.score;
    });

    // (b) Prefix matches
    const prefixMatches = this.trie.prefixSearch(normQuery, maxSuggestions * 6);
    for (const [c, freq] of prefixMatches) {
      const dist = damerauLevenshteinDistance(normQuery, c);
      if (dist <= maxDistance) {
        const score = similarityRatio(normQuery, c);
        if (score >= minScore) {
          prefixCands.push({
            word: c,
            score,
            distance: dist,
            frequency: freq,
            match_type: dist === 0 ? 'exact_match' : 'prefix',
          });
        }
      }
    }
    prefixCands.sort((a, b) => {
      if (a.distance !== b.distance) return a.distance - b.distance;
      if (b.frequency !== a.frequency) return b.frequency - a.frequency;
      return b.score - a.score;
    });

    // (c) Spelling typo matches
    const deletesCands = this.getCandidates(normQuery, maxDistance);
    for (const c of deletesCands) {
      const dist = damerauLevenshteinDistance(normQuery, c);
      if (dist <= maxDistance) {
        const score = similarityRatio(normQuery, c);
        if (score >= minScore) {
          const freq = this.wordFreq[c] ?? 1.0;
          spellCands.push({
            word: c,
            score,
            distance: dist,
            frequency: freq,
            match_type: 'spelling',
          });
        }
      }
    }
    spellCands.sort((a, b) => {
      if (a.distance !== b.distance) return a.distance - b.distance;
      if (b.frequency !== a.frequency) return b.frequency - a.frequency;
      return b.score - a.score;
    });

    const activeSources = [suffixCands, prefixCands, spellCands].filter(
      (s) => s.length > 0
    );
    const results: SuggestionItem[] = [];
    let idx = 0;

    while (
      results.length < maxSuggestions &&
      activeSources.some((s) => idx < s.length)
    ) {
      for (const s of activeSources) {
        if (idx < s.length) {
          const item = s[idx]!;
          if (!seen.has(item.word)) {
            seen.add(item.word);
            results.push(item);
            if (results.length >= maxSuggestions) break;
          }
        }
      }
      idx++;
    }

    return results;
  }

  public fuzzySearch(
    query: string,
    maxSuggestions: number = 10,
    minScore: number = 0.4,
    maxDistance: number = 5
  ): SuggestionItem[] {
    return this.suggest(query, maxSuggestions, maxDistance, minScore, 'fuzzy');
  }
}

// Global default singleton instance (lazy on first request)
export const spellChecker = new KhmerSpellChecker();
