import * as tf from '@tensorflow/tfjs-core';
import * as tflite from '@tensorflow/tfjs-tflite';
import { segKcc } from './kccs';
import { TextGraphEncoder } from './preprocessor';

export interface SegmenterOptions {
  modelUrl: string | ArrayBuffer;
  vocabUrlOrData: string | string[];
  wasmPath?: string;
}

export class Segmenter {
  private encoder: TextGraphEncoder | null = null;
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
      const res = await fetch(options.vocabUrlOrData);
      vocab = await res.json();
    }

    this.encoder = new TextGraphEncoder(vocab);

    // Load TFLite Model
    if (typeof options.modelUrl === 'string') {
      this.tfliteModel = await tflite.loadTFLiteModel(options.modelUrl);
    } else {
      this.tfliteModel = await tflite.loadTFLiteModel(options.modelUrl);
    }
  }

  async segment(text: string): Promise<string[]> {
    if (!this.tfliteModel || !this.encoder) {
      throw new Error('Segmenter is not initialized. Call Segmenter.create(...) first.');
    }

    const cleanedStr = text.replace(/\u200b/g, ' ').trim();
    if (!cleanedStr) {
      return [];
    }

    const kccs = segKcc(cleanedStr);
    const graph = this.encoder.encode(kccs, 0);

    const edgeIndicesTensor = graph.edgeIndices.length > 0
      ? tf.tensor2d(graph.edgeIndices, [graph.edgeIndices.length, 2], 'int32')
      : tf.zeros([0, 2], 'int32');

    const inputs = {
      'char_indices': tf.tensor1d(graph.charIndices, 'int32'),
      'char_to_node_map': tf.tensor1d(graph.charToNodeMap, 'int32'),
      'node_types': tf.tensor1d(graph.nodeTypes, 'int32'),
      'edge_indices': edgeIndicesTensor,
      'edge_types': tf.tensor1d(graph.edgeTypes, 'int32'),
      'batch_ids': tf.zeros([graph.nodeTypes.length], 'int32')
    };

    const outputTensors = this.tfliteModel.predict(inputs);
    const rawOutput = Object.values(outputTensors)[0] as tf.Tensor;
    const argmaxOutput = rawOutput.argMax(-1);
    const yPred = await argmaxOutput.data();

    // Cleanup Tensors from memory
    Object.values(inputs).forEach(t => t.dispose());
    rawOutput.dispose();
    argmaxOutput.dispose();

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
    return formattedStr.split(' ');
  }
}
