import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs-core';
import * as tflite from '@tensorflow/tfjs-tflite';
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
} from './kccs';
import type { GraphData } from './preprocessor';
import { TextGraphEncoder } from './preprocessor';
import { GLNNModel, GLNNWeights } from './glnn';

export {
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
  EN_CHARS,
  TextGraphEncoder,
  GLNNModel
};
export type { GraphData, GLNNWeights };
export interface SegmenterOptions {
  modelUrl: string | ArrayBuffer;
  vocabUrlOrData: string | string[];
  wasmPath?: string;
}

export interface SegmentProgress {
  current: number;
  total: number;
  percentage: number;
}

export interface SegmentOptions {
  maxKccsPerChunk?: number;
  onProgress?: (progress: SegmentProgress) => void;
}

export function chunkKccs(kccs: string[], maxChunkSize: number = 60): string[][] {
  if (kccs.length <= maxChunkSize) return [kccs];

  const chunks: string[][] = [];
  let current: string[] = [];

  for (let i = 0; i < kccs.length; i++) {
    const kcc = kccs[i];
    current.push(kcc);

    const isSentenceEnd = kcc === '។' || kcc === '៕' || kcc === '?' || kcc === '!' || kcc.includes('\n');
    const isSpace = kcc === ' ';

    if (current.length >= maxChunkSize || (current.length >= 25 && (isSentenceEnd || isSpace))) {
      chunks.push(current);
      current = [];
    }
  }

  if (current.length > 0) {
    chunks.push(current);
  }

  return chunks;
}

export function patchTFLiteDynamicShapes(buffer: ArrayBuffer): ArrayBuffer {
  const bytes = new Uint8Array(buffer.slice(0));
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

  if (view.byteLength < 24) return bytes.buffer;

  const rootTableOffset = view.getInt32(0, true);
  if (rootTableOffset < 0 || rootTableOffset + 4 > view.byteLength) return bytes.buffer;

  const vtableOffset = rootTableOffset - view.getInt32(rootTableOffset, true);
  if (vtableOffset < 0 || vtableOffset + 10 > view.byteLength) return bytes.buffer;

  const subgraphsFieldOffset = view.getUint16(vtableOffset + 8, true);
  if (subgraphsFieldOffset === 0) return bytes.buffer;

  const subgraphsVecOffset = rootTableOffset + subgraphsFieldOffset + view.getInt32(rootTableOffset + subgraphsFieldOffset, true);
  const numSubgraphs = view.getInt32(subgraphsVecOffset, true);
  if (numSubgraphs <= 0) return bytes.buffer;

  const subgraphOffset = subgraphsVecOffset + 4 + view.getInt32(subgraphsVecOffset + 4, true);
  const sgVtableOffset = subgraphOffset - view.getInt32(subgraphOffset, true);

  const tensorsFieldOffset = view.getUint16(sgVtableOffset + 4, true);
  const inputsFieldOffset = view.getUint16(sgVtableOffset + 6, true);
  if (tensorsFieldOffset === 0 || inputsFieldOffset === 0) return bytes.buffer;

  const tensorsVecOffset = subgraphOffset + tensorsFieldOffset + view.getInt32(subgraphOffset + tensorsFieldOffset, true);
  const inputsVecOffset = subgraphOffset + inputsFieldOffset + view.getInt32(subgraphOffset + inputsFieldOffset, true);
  const numInputs = view.getInt32(inputsVecOffset, true);

  for (let i = 0; i < numInputs; i++) {
    const tensorIdx = view.getInt32(inputsVecOffset + 4 + i * 4, true);
    const tensorOffset = tensorsVecOffset + 4 + tensorIdx * 4 + view.getInt32(tensorsVecOffset + 4 + tensorIdx * 4, true);
    const tVtableOffset = tensorOffset - view.getInt32(tensorOffset, true);
    const shapeField = view.getUint16(tVtableOffset + 4, true);

    if (shapeField > 0) {
      const shapeVecOffset = tensorOffset + shapeField + view.getInt32(tensorOffset + shapeField, true);
      const shapeLen = view.getInt32(shapeVecOffset, true);
      if (shapeLen > 0) {
        // Set dynamic sequence dimension to -1
        view.setInt32(shapeVecOffset + 4, -1, true);
      }
    }
  }

  return bytes.buffer;
}

export class Segmenter {
  private encoder: TextGraphEncoder | null = null;
  private glnnModel: GLNNModel | null = null;
  private tfliteModel: tflite.TFLiteModel | null = null;

  static async create(options: SegmenterOptions): Promise<Segmenter> {
    const segmenter = new Segmenter();
    await segmenter.init(options);
    return segmenter;
  }

  async init(options: SegmenterOptions): Promise<void> {
    if (options.wasmPath) {
      tflite.setWasmPath(options.wasmPath);
    }

    // Load Vocab
    let vocab: string[];
    if (Array.isArray(options.vocabUrlOrData)) {
      vocab = options.vocabUrlOrData;
    } else {
      try {
        const res = await fetch(options.vocabUrlOrData);
        vocab = await res.json();
      } catch {
        vocab = [];
      }
    }

    this.encoder = new TextGraphEncoder(vocab);

    // Load Model
    let modelData: ArrayBuffer | null = null;
    if (typeof options.modelUrl !== 'string') {
      modelData = options.modelUrl;
    } else {
      try {
        const res = await fetch(options.modelUrl);
        if (res.ok) {
          modelData = await res.arrayBuffer();
        }
      } catch {
        // Fallback for mocks / node paths
      }
    }

    if (modelData && modelData.byteLength > 100) {
      try {
        this.glnnModel = GLNNModel.fromFlatBuffer(modelData);
      } catch (err) {
        console.warn('Falling back to TFLite model runner:', err);
      }
    }

    if (!this.glnnModel) {
      this.tfliteModel = await tflite.loadTFLiteModel(options.modelUrl);
    }
  }
  async segment(text: string, options?: SegmentOptions): Promise<string[]> {
    if ((!this.glnnModel && !this.tfliteModel) || !this.encoder) {
      throw new Error('Segmenter is not initialized. Call Segmenter.create(...) first.');
    }

    const cleanedStr = text.trim();
    if (!cleanedStr) {
      return [];
    }

    const allKccs = segKcc(cleanedStr);
    if (allKccs.length === 0) {
      return [];
    }

    const maxChunkSize = options?.maxKccsPerChunk ?? 60;
    const chunks = chunkKccs(allKccs, maxChunkSize);
    const allTokens: string[] = [];

    for (let c = 0; c < chunks.length; c++) {
      const kccs = chunks[c];
      if (options?.onProgress) {
        options.onProgress({
          current: c + 1,
          total: chunks.length,
          percentage: Math.round(((c + 1) / chunks.length) * 100)
        });
      }

      // Yield to UI event loop if multiple chunks
      if (chunks.length > 1) {
        const { promise, resolve } = Promise.withResolvers<void>();
        setTimeout(resolve, 0);
        await promise;
      }
      const graph = this.encoder.encode(kccs, 0);
      let yPred: ArrayLike<number>;

      if (this.glnnModel) {
        yPred = this.glnnModel.predict(graph);
      } else if (this.tfliteModel) {
        const edgeIndicesTensor = graph.edgeIndices.length > 0
          ? tf.tensor2d(graph.edgeIndices, [graph.edgeIndices.length, 2], 'int32')
          : tf.zeros([0, 2], 'int32');

        const baseInputs: Record<string, tf.Tensor> = {
          'char_indices': tf.tensor1d(graph.charIndices, 'int32'),
          'char_to_node_map': tf.tensor1d(graph.charToNodeMap, 'int32'),
          'node_types': tf.tensor1d(graph.nodeTypes, 'int32'),
          'edge_indices': edgeIndicesTensor,
          'edge_types': tf.tensor1d(graph.edgeTypes, 'int32'),
          'batch_ids': tf.zeros([graph.nodeTypes.length], 'int32')
        };

        const expectedInputs = this.tfliteModel.inputs;
        const inputs: Record<string, tf.Tensor> = {};

        if (expectedInputs && expectedInputs.length > 0) {
          for (const info of expectedInputs) {
            const fullName = info.name;
            const cleanName = fullName.replace(/^serving_default_/, '').replace(/:\d+$/, '');
            if (baseInputs[cleanName]) {
              inputs[fullName] = baseInputs[cleanName];
            } else if (baseInputs[fullName]) {
              inputs[fullName] = baseInputs[fullName];
            }
          }
        } else {
          Object.assign(inputs, baseInputs);
        }

        const outputTensors = this.tfliteModel.predict(inputs);
        const rawOutput = Object.values(outputTensors)[0] as tf.Tensor;
        const argmaxOutput = tf.argMax(rawOutput, -1);
        yPred = await argmaxOutput.data();
        tf.dispose([inputs, outputTensors, argmaxOutput]);
      } else {
        throw new Error('No model loaded.');
      }

      let formattedStr = '';
      kccs.forEach((chunk, i) => {
        const pred = yPred[i];
        if (pred === 0 || pred === 3) {
          formattedStr += ` ${chunk}`;
        } else {
          formattedStr += chunk;
        }
      });

      formattedStr = formattedStr.trim().replace(/\s+/g, ' ');
      const chunkTokens = formattedStr.split(' ').filter(Boolean);
      allTokens.push(...chunkTokens);
    }

    return allTokens;
  }
}
