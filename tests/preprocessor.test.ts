import { describe, it, expect } from 'vitest';
import { TextGraphEncoder } from '../src/preprocessor';

describe('TextGraphEncoder', () => {
  const vocab = ['<UNK/>', 'ខ', '្', 'ញ', 'ុ', 'ំ'];
  const encoder = new TextGraphEncoder(vocab, true);

  it('exposes vocabSize getter correctly', () => {
    expect(encoder.vocabSize).toBe(6);
  });

  it('encodes KCCs into valid graph data (directed)', () => {
    const kccs = ['ខ្ញុំ', 'ខ'];
    const graph = encoder.encode(kccs);

    expect(graph.nodeTypes).toEqual([0, 0]);
    // Edge between node 0 and node 1:
    // eKccNext: [[1, 0]] (type 0)
    // eKccPrev: [[0, 1]] (type 1)
    expect(graph.edgeIndices).toEqual([[1, 0], [0, 1]]);
    expect(graph.edgeTypes).toEqual([0, 1]);
    expect(graph.charToNodeMap).toEqual([0, 0, 0, 0, 0, 1]);
  });

  it('encodes undirected edges with uniform edge types', () => {
    const undirectedEncoder = new TextGraphEncoder(vocab, false);
    const kccs = ['ខ្ញុំ', 'ខ'];
    const graph = undirectedEncoder.encode(kccs);

    expect(graph.edgeIndices).toEqual([[1, 0], [0, 1]]);
    expect(graph.edgeTypes).toEqual([0, 0]);
  });

  it('handles single KCC and empty KCCs correctly', () => {
    const singleGraph = encoder.encode(['ខ្ញុំ']);
    expect(singleGraph.nodeTypes).toEqual([0]);
    expect(singleGraph.edgeIndices).toEqual([]);
    expect(singleGraph.edgeTypes).toEqual([]);
    expect(singleGraph.charToNodeMap).toEqual([0, 0, 0, 0, 0]);

    const emptyGraph = encoder.encode([]);
    expect(emptyGraph.nodeTypes).toEqual([]);
    expect(emptyGraph.edgeIndices).toEqual([]);
    expect(emptyGraph.edgeTypes).toEqual([]);
    expect(emptyGraph.charIndices).toEqual([]);
    expect(emptyGraph.charToNodeMap).toEqual([]);
  });

  it('maps unknown characters to <UNK/> index', () => {
    const kccs = ['ក']; // 'ក' is not in vocab
    const graph = encoder.encode(kccs);
    expect(graph.charIndices).toEqual([0]); // maps to unkIndex (0)
  });

  it('supports nodeOffset parameter', () => {
    const kccs = ['ខ', 'ញ'];
    const graph = encoder.encode(kccs, 5);

    expect(graph.edgeIndices).toEqual([[6, 5], [5, 6]]);
    expect(graph.charToNodeMap).toEqual([5, 6]);
  });
});
