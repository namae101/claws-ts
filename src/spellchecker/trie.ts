export class TrieNode {
  public children: Record<string, TrieNode> = {};
  public isEnd: boolean = false;
  public freq: number = 0.0;
}

export class PrefixTrie {
  public root: TrieNode = new TrieNode();

  public clear(): void {
    this.root = new TrieNode();
  }

  public insert(word: string, freq: number = 1.0): void {
    let curr = this.root;
    for (let i = 0; i < word.length; i++) {
      const char = word[i]!;
      if (!curr.children[char]) {
        curr.children[char] = new TrieNode();
      }
      curr = curr.children[char]!;
    }
    curr.isEnd = true;
    curr.freq = Math.max(curr.freq, freq);
  }

  public prefixSearch(
    prefix: string,
    maxResults: number = 10
  ): Array<[word: string, freq: number]> {
    let curr = this.root;
    for (let i = 0; i < prefix.length; i++) {
      const char = prefix[i]!;
      if (!curr.children[char]) {
        return [];
      }
      curr = curr.children[char]!;
    }

    const matches: Array<[word: string, freq: number]> = [];

    const dfs = (node: TrieNode, path: string): void => {
      if (node.isEnd) {
        matches.push([path, node.freq]);
      }
      const keys = Object.keys(node.children);
      for (const char of keys) {
        dfs(node.children[char]!, path + char);
      }
    };

    dfs(curr, prefix);

    // Sort by frequency descending, then length ascending, then alphabetical
    matches.sort((a, b) => {
      if (b[1] !== a[1]) return b[1] - a[1];
      if (a[0].length !== b[0].length) return a[0].length - b[0].length;
      return a[0].localeCompare(b[0]);
    });

    return matches.slice(0, maxResults);
  }
}
