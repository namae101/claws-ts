/**
 * Check if the text contains any Khmer Unicode characters (U+1780..U+17FF or U+19E0..U+19FF).
 */
export function isKhmerText(text: string): boolean {
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    if (
      (code >= 0x1780 && code <= 0x17ff) ||
      (code >= 0x19e0 && code <= 0x19ff)
    ) {
      return true;
    }
  }
  return false;
}

/**
 * Compute the Damerau-Levenshtein edit distance between two strings.
 * Handles insertions, deletions, substitutions, and adjacent transpositions.
 */
export function damerauLevenshteinDistance(s1: string, s2: string): number {
  if (s1 === s2) return 0;
  const len1 = s1.length;
  const len2 = s2.length;
  if (len1 === 0) return len2;
  if (len2 === 0) return len1;

  const d: number[][] = Array.from({ length: len1 + 1 }, () =>
    new Array<number>(len2 + 1).fill(0)
  );

  for (let i = 0; i <= len1; i++) {
    d[i]![0] = i;
  }
  for (let j = 0; j <= len2; j++) {
    d[0]![j] = j;
  }

  for (let i = 1; i <= len1; i++) {
    for (let j = 1; j <= len2; j++) {
      const cost = s1[i - 1] === s2[j - 1] ? 0 : 1;
      d[i]![j] = Math.min(
        d[i - 1]![j]! + 1, // deletion
        d[i]![j - 1]! + 1, // insertion
        d[i - 1]![j - 1]! + cost // substitution
      );

      // Transposition check
      if (
        i > 1 &&
        j > 1 &&
        s1[i - 1] === s2[j - 2] &&
        s1[i - 2] === s2[j - 1]
      ) {
        d[i]![j] = Math.min(d[i]![j]!, d[i - 2]![j - 2]! + 1);
      }
    }
  }

  return d[len1]![len2]!;
}

export const levenshteinDistance = damerauLevenshteinDistance;

/**
 * Calculate normalized fuzzy similarity score between 0.0 and 1.0.
 * 1.0 means exact match, 0.0 means completely different.
 */
export function similarityRatio(s1: string, s2: string): number {
  if (s1 === s2) return 1.0;
  const maxLen = Math.max(s1.length, s2.length);
  if (maxLen === 0) return 1.0;
  const dist = damerauLevenshteinDistance(s1, s2);
  const ratio = Math.max(0.0, 1.0 - dist / maxLen);
  return Math.round(ratio * 10000) / 10000;
}
