import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as tf from '@tensorflow/tfjs-core';
import * as tflite from '@tensorflow/tfjs-tflite';
import {
  Segmenter,
  TextGraphEncoder,
  segKcc,
  isKhmerChar,
  isStartOfKcc
} from '../src/index';

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
  });
});
