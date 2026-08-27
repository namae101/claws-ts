import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as tf from '@tensorflow/tfjs-core';
import * as tflite from '@tensorflow/tfjs-tflite';
import {
  Segmenter,
  TextGraphEncoder,
  segKcc,
  isKhmerChar,
  isStartOfKcc,
  patchTFLiteDynamicShapes,
  GLNNModel,
  chunkKccs,
  normalize,
  spellChecker,
  KhmerSpellChecker
} from '../src/index';
import * as fs from 'node:fs';
import * as path from 'node:path';

vi.mock('@tensorflow/tfjs-tflite', () => ({
  loadTFLiteModel: vi.fn(),
  setWasmPath: vi.fn()
}));

describe('Segmenter', () => {
  const sampleVocab = ['<UNK/>', 'ខ', '្', 'ញ', 'ុ', 'ំ', 'ស', 'រ', 'ឡ', 'ា', 'ញ', '់', 'ប', 'ទ', 'េ', 'ក', 'ព', 'ជ'];

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('module exports', () => {
    it('exports core classes and utilities', () => {
      expect(Segmenter).toBeDefined();
      expect(TextGraphEncoder).toBeDefined();
      expect(typeof segKcc).toBe('function');
      expect(typeof isKhmerChar).toBe('function');
      expect(typeof isStartOfKcc).toBe('function');
      expect(typeof patchTFLiteDynamicShapes).toBe('function');
      expect(typeof chunkKccs).toBe('function');
    });
    it('safely handles short or invalid buffers in patchTFLiteDynamicShapes', () => {
      const smallBuf = new ArrayBuffer(8);
      const result = patchTFLiteDynamicShapes(smallBuf);
      expect(result.byteLength).toBe(8);
    });
  });

  describe('uninitialized behavior', () => {
    it('throws error if segment is called before initialization', async () => {
      const segmenter = new Segmenter();
      await expect(segmenter.segment('កម្ពុជា')).rejects.toThrowError(
        'Segmenter is not initialized. Call Segmenter.create(...) first.'
      );
    });
  });

  describe('segmentation with model inference', () => {
    it('initializes with wasmPath and array vocab', async () => {
      vi.mocked(tflite.loadTFLiteModel).mockResolvedValue({
        predict: vi.fn()
      } as unknown as tflite.TFLiteModel);

      const segmenter = await Segmenter.create({
        modelUrl: '/models/model.tflite',
        vocabUrlOrData: sampleVocab,
        wasmPath: 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite/dist/'
      });

      expect(tflite.setWasmPath).toHaveBeenCalledWith('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite/dist/');
      expect(tflite.loadTFLiteModel).toHaveBeenCalledWith('/models/model.tflite');
      expect(segmenter).toBeInstanceOf(Segmenter);
    });

    it('returns empty array for empty or whitespace-only input without running model', async () => {
      const predictFn = vi.fn();
      vi.mocked(tflite.loadTFLiteModel).mockResolvedValue({
        predict: predictFn
      } as unknown as tflite.TFLiteModel);

      const segmenter = await Segmenter.create({
        modelUrl: 'dummy.tflite',
        vocabUrlOrData: sampleVocab
      });

      expect(await segmenter.segment('')).toEqual([]);
      expect(await segmenter.segment('   ')).toEqual([]);
      expect(predictFn).not.toHaveBeenCalled();
    });

    it('correctly segments Khmer text into words based on GNN output predictions', async () => {
      // "ខ្ញុំស្រឡាញ់ប្រទេសកម្ពុជា" -> 10 KCCs:
      // ['ខ្ញុំ', 'ស្រ', 'ឡា', 'ញ់', 'ប្រ', 'ទេ', 'ស', 'ក', 'ម្ពុ', 'ជា']
      // Desired words: ["ខ្ញុំ", "ស្រឡាញ់", "ប្រទេស", "កម្ពុជា"]
      // Boundary tags:
      // ខ្ញុំ -> tag 0 (start)
      // ស្រ -> tag 0 (start), ឡា -> tag 1, ញ់ -> tag 1
      // ប្រ -> tag 0 (start), ទេ -> tag 1, ស -> tag 1
      // ក -> tag 3 (start / single), ម្ពុ -> tag 1, ជា -> tag 1
      const expectedTags = [0, 0, 1, 1, 0, 1, 1, 3, 1, 1];

      const mockPredict = vi.fn().mockImplementation((inputs: Record<string, tf.Tensor>) => {
        // Create 1-hot / logits tensor for the tags [numKccs, 4]
        const numNodes = inputs['node_types'].shape[0];
        const logits: number[][] = [];
        for (let i = 0; i < numNodes; i++) {
          const row = [0, 0, 0, 0];
          const tag = expectedTags[i] ?? 1;
          row[tag] = 10.0; // high logit for the predicted tag
          logits.push(row);
        }
        return {
          output: tf.tensor2d(logits, [numNodes, 4])
        };
      });

      vi.mocked(tflite.loadTFLiteModel).mockResolvedValue({
        predict: mockPredict
      } as unknown as tflite.TFLiteModel);

      const segmenter = await Segmenter.create({
        modelUrl: 'dummy.tflite',
        vocabUrlOrData: sampleVocab
      });

      const initialTensors = tf.memory().numTensors;
      const text = 'ខ្ញុំស្រឡាញ់ប្រទេសកម្ពុជា';
      const words = await segmenter.segment(text);

      expect(words).toEqual(['ខ្ញុំ', 'ស្រឡាញ់', 'ប្រទេស', 'កម្ពុជា']);
      expect(mockPredict).toHaveBeenCalledTimes(1);

      // Verify no tensor memory leaks
      const finalTensors = tf.memory().numTensors;
      expect(finalTensors).toBe(initialTensors);
    });

    it('handles single-word text segmentation', async () => {
      const mockPredict = vi.fn().mockImplementation((inputs: Record<string, tf.Tensor>) => {
        const numNodes = inputs['node_types'].shape[0];
        // For 'កម្ពុជា' (3 KCCs: 'ក', 'ម្ពុ', 'ជា'), tags: [0, 1, 1]
        const logits = [
          [10, 0, 0, 0],
          [0, 10, 0, 0],
          [0, 10, 0, 0]
        ];
        return {
          output: tf.tensor2d(logits, [numNodes, 4])
        };
      });

      vi.mocked(tflite.loadTFLiteModel).mockResolvedValue({
        predict: mockPredict
      } as unknown as tflite.TFLiteModel);

      const segmenter = await Segmenter.create({
        modelUrl: 'dummy.tflite',
        vocabUrlOrData: sampleVocab
      });

      const words = await segmenter.segment('កម្ពុជា');
      expect(words).toEqual(['កម្ពុជា']);
    });

    it('dynamically matches serving_default_*:0 input names from model metadata', async () => {
      let receivedInputKeys: string[] = [];
      const mockPredict = vi.fn().mockImplementation((inputs: Record<string, tf.Tensor>) => {
        receivedInputKeys = Object.keys(inputs);
        const numNodes = inputs['serving_default_node_types:0'].shape[0];
        return {
          output: tf.tensor2d([[10, 0, 0, 0]], [numNodes, 4])
        };
      });

      vi.mocked(tflite.loadTFLiteModel).mockResolvedValue({
        inputs: [
          { name: 'serving_default_char_indices:0' },
          { name: 'serving_default_char_to_node_map:0' },
          { name: 'serving_default_node_types:0' },
          { name: 'serving_default_edge_indices:0' },
          { name: 'serving_default_edge_types:0' },
          { name: 'serving_default_batch_ids:0' }
        ],
        predict: mockPredict
      } as unknown as tflite.TFLiteModel);

      const segmenter = await Segmenter.create({
        modelUrl: 'dummy.tflite',
        vocabUrlOrData: sampleVocab
      });

      const words = await segmenter.segment('ក');
      expect(words).toEqual(['ក']);
      expect(receivedInputKeys).toEqual([
        'serving_default_char_indices:0',
        'serving_default_char_to_node_map:0',
        'serving_default_node_types:0',
        'serving_default_edge_indices:0',
        'serving_default_edge_types:0',
        'serving_default_batch_ids:0'
      ]);
    });
  });

  describe('end-to-end with GLNN model FlatBuffer', () => {
    it('segments real Khmer phrases using GLNNModel from model.tflite', async () => {
      const modelPath = path.resolve('website/public/models/model.tflite');
      const vocabPath = path.resolve('website/public/models/vocab.json');
      const modelBuf = fs.readFileSync(modelPath).buffer;
      const vocab = JSON.parse(fs.readFileSync(vocabPath, 'utf8'));

      const segmenter = await Segmenter.create({
        modelUrl: modelBuf,
        vocabUrlOrData: vocab
      });

      const words = await segmenter.segment('ខ្ញុំស្រឡាញ់ប្រទេសកម្ពុជា');
      expect(words).toEqual(['ខ្ញុំ', 'ស្រឡាញ់', 'ប្រទេស', 'កម្ពុជា']);

      const words2 = await segmenter.segment('សាកលវិទ្យាល័យភូមិន្ទភ្នំពេញ');
      expect(words2).toEqual(['សាកលវិទ្យាល័យ', 'ភូមិន្ទ', 'ភ្នំពេញ']);
    });

    it('splits large inputs into chunks and calls onProgress callback', async () => {
      const modelPath = path.resolve('website/public/models/model.tflite');
      const vocabPath = path.resolve('website/public/models/vocab.json');
      const modelBuf = fs.readFileSync(modelPath).buffer;
      const vocab = JSON.parse(fs.readFileSync(vocabPath, 'utf8'));

      const segmenter = await Segmenter.create({
        modelUrl: modelBuf,
        vocabUrlOrData: vocab
      });

      const longText = 'សាកលវិទ្យាល័យភូមិន្ទភ្នំពេញ គឺជាគ្រឹះស្ថានឧត្តមសិក្សាដ៏ចំណាស់ជាងគេនៅកម្ពុជា។ ស្ថាប័ននេះត្រូវបានបង្កើតឡើងនៅក្នុងឆ្នាំ១៩៦០ និងមានតួនាទីយ៉ាងសំខាន់ក្នុងការបណ្តុះបណ្តាលធនធានមនុស្ស។';
      const progressReports: Array<{ current: number; total: number; percentage: number }> = [];

      const words = await segmenter.segment(longText, {
        maxKccsPerChunk: 25,
        onProgress: (p) => progressReports.push(p)
      });

      expect(words.length).toBeGreaterThan(10);
      expect(progressReports.length).toBeGreaterThan(1);
      expect(progressReports[progressReports.length - 1].percentage).toBe(100);
    });
  });

  describe('Unicode Normalization (betterkhmer)', () => {
    it('normalizes out-of-order Khmer vowels and subscripts', () => {
      expect(normalize('ខែ្មរ')).toBe('ខ្មែរ');
      expect(normalize('ខ្មែរ')).toBe('ខ្មែរ');
    });

    it('removes invisible zero-width spaces', () => {
      expect(normalize('ខ្ញុំ\u200Bស្រឡាញ់\u200Cកម្ពុជា\uFEFF')).toBe('ខ្ញុំស្រឡាញ់កម្ពុជា');
    });

    it('replaces Khmer punctuation and symbols with regular space', () => {
      expect(normalize('កម្ពុជា។ ស្រឡាញ់៕ សួស្តី៖ បាទ៚')).toBe('កម្ពុជា ស្រឡាញ់ សួស្តី បាទ ');
    });
  });

  describe('Spell Checking & Multi-Strategy Suggestions', () => {
    it('initializes dictionary and checks correct and misspelled words', async () => {
      await spellChecker.init();

      expect(spellChecker.getWordCount()).toBeGreaterThan(1000);

      // Correct word
      const correctResult = spellChecker.checkWord('ខ្ញុំ');
      expect(correctResult.is_correct).toBe(true);
      expect(correctResult.suggestions).toEqual([]);

      // Misspelled word
      const misspelledResult = spellChecker.checkWord('ម៉ាល់', 5);
      expect(misspelledResult.is_correct).toBe(false);
      expect(misspelledResult.suggestions.length).toBeGreaterThan(0);

      // Non-word / numbers / punctuation
      expect(spellChecker.checkWord('123').is_correct).toBe(true);
      expect(spellChecker.checkWord('។').is_correct).toBe(true);
    });

    it('finds suffix suggestions via Reverse Suffix Trie', async () => {
      await spellChecker.init();
      const suffixes = spellChecker.suggestSuffix('វិទ្យា', 5);
      expect(suffixes.length).toBeGreaterThan(0);
      expect(suffixes[0].word).toBe('វិទ្យា');
      expect(suffixes.some(s => s.word.endsWith('វិទ្យា'))).toBe(true);
    });

    it('finds prefix suggestions via Prefix Trie', async () => {
      await spellChecker.init();
      const prefixes = spellChecker.suggestPrefix('កម្ពុ', 5);
      expect(prefixes.length).toBeGreaterThan(0);
      expect(prefixes.some(p => p.word.startsWith('កម្ពុ'))).toBe(true);
    });
  });

  describe('Khmer-aware chunking by space & punctuation', () => {
    it('closes chunks at Khmer sentence punctuation regardless of size', () => {
      const kccs = segKcc('សួស្តី! តើអ្នកសុខសប្បាយទេ? សូមអរគុណ។');
      const chunks = chunkKccs(kccs, 60);

      // 3 sentences (ending in !, ?, ។) -> exactly 3 chunks
      expect(chunks.length).toBe(3);
      expect(chunks[0]!.join('').endsWith('!')).toBe(true);
      expect(chunks.some(c => c.join('').endsWith('?'))).toBe(true);
      expect(chunks.some(c => c.join('').endsWith('។'))).toBe(true);
      // No chunk exceeds the hard cap
      for (const c of chunks) expect(c.length).toBeLessThanOrEqual(60);
    });

    it('batches short phrases together instead of splitting on every space', () => {
      // 10 space-separated words: chunker batches them rather than creating 10 single-word chunks
      const kccs = segKcc('ខ្ញុំ ស្រឡាញ់ ភាសា ខ្មែរ ព្រោះ វា ជា ភាសា មាត់ របស់ ខ្ញុំ');
      const chunks = chunkKccs(kccs, 60, 25);
      expect(chunks.length).toBeLessThanOrEqual(2);
      expect(chunks.flat().join('')).toBe(kccs.join(''));
    });

    it('hard-splits unbroken runs longer than maxChunkSize', () => {
      const kccs = segKcc('ក'.repeat(200));
      const chunks = chunkKccs(kccs, 60);
      expect(chunks.length).toBeGreaterThan(1);
      for (const c of chunks) expect(c.length).toBeLessThanOrEqual(60);
    });
  });
});
