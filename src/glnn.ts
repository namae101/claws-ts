import * as tf from '@tensorflow/tfjs-core';
import type { GraphData } from './preprocessor';

interface RGCNLayerWeights {
  gather: tf.Tensor4D;
  matmul: tf.Tensor2D;
  add: tf.Tensor1D;
  bn_mul: tf.Tensor1D | null;
  bn_add: tf.Tensor1D | null;
}

export interface GLNNWeights {
  charEmb: tf.Tensor2D;
  bn0_mul: tf.Tensor1D;
  bn0_add: tf.Tensor1D;
  layers: RGCNLayerWeights[];
  out_w: tf.Tensor2D;
  out_b: tf.Tensor1D;
}

export class GLNNModel {
  private weights: GLNNWeights;

  constructor(weights: GLNNWeights) {
    this.weights = weights;
  }

  static fromFlatBuffer(buffer: ArrayBuffer): GLNNModel {
    const bytes = new Uint8Array(buffer);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

    const rootTableOffset = view.getInt32(0, true);
    const subgraphsVecOffset = rootTableOffset + 20 + view.getInt32(rootTableOffset + 20, true);
    const subgraphOffset = subgraphsVecOffset + 4 + view.getInt32(subgraphsVecOffset + 4, true);
    const tensorsVecOffset = subgraphOffset + 20 + view.getInt32(subgraphOffset + 20, true);
    const buffersVecOffset = rootTableOffset + 12 + view.getInt32(rootTableOffset + 12, true);

    function getBufferBytes(bufIdx: number): Uint8Array {
      const bufElemOffset = buffersVecOffset + 4 + bufIdx * 4;
      const tableOffset = bufElemOffset + view.getInt32(bufElemOffset, true);
      const vtableOffset = tableOffset - view.getInt32(tableOffset, true);
      const dataFieldOffset = view.getUint16(vtableOffset + 4, true);
      if (dataFieldOffset === 0) return new Uint8Array(0);
      const dataVecOffset = tableOffset + dataFieldOffset + view.getInt32(tableOffset + dataFieldOffset, true);
      const dataLen = view.getInt32(dataVecOffset, true);
      return bytes.subarray(dataVecOffset + 4, dataVecOffset + 4 + dataLen);
    }

    function getTensorScale(t: number): number {
      const tensorOffset = tensorsVecOffset + 4 + t * 4 + view.getInt32(tensorsVecOffset + 4 + t * 4, true);
      const quantOffset = tensorOffset + 8 + view.getInt32(tensorOffset + 8, true);
      const qVtableOffset = quantOffset - view.getInt32(quantOffset, true);
      const scaleFieldOffset = view.getUint16(qVtableOffset + 8, true);
      if (scaleFieldOffset === 0) return 1.0;
      const scaleVecOffset = quantOffset + scaleFieldOffset + view.getInt32(quantOffset + scaleFieldOffset, true);
      return view.getFloat32(scaleVecOffset + 4, true);
    }

    function getDequantizedTensor<T extends tf.Tensor>(tensorIdx: number, bufIdx: number, shape: number[]): T {
      const b = getBufferBytes(bufIdx);
      const scale = getTensorScale(tensorIdx);
      const int8 = new Int8Array(b.buffer, b.byteOffset, b.length);
      const float32 = new Float32Array(int8.length);
      for (let i = 0; i < int8.length; i++) {
        float32[i] = int8[i] * scale;
      }
      return tf.tensor(float32, shape, 'float32') as T;
    }

    function getFloatTensor<T extends tf.Tensor>(bufIdx: number, shape: number[]): T {
      const b = getBufferBytes(bufIdx);
      const float32 = new Float32Array(b.buffer, b.byteOffset, b.length / 4);
      return tf.tensor(float32, shape, 'float32') as T;
    }

    const charEmb = getDequantizedTensor<tf.Tensor2D>(36, 37, [147, 512]);
    const bn0_mul = getFloatTensor<tf.Tensor1D>(17, [512]);
    const bn0_add = getFloatTensor<tf.Tensor1D>(18, [512]);

    const rgcn0: RGCNLayerWeights = {
      gather: getDequantizedTensor<tf.Tensor4D>(35, 36, [2, 4, 128, 128]),
      matmul: getDequantizedTensor<tf.Tensor2D>(37, 38, [512, 512]),
      add: getFloatTensor<tf.Tensor1D>(13, [512]),
      bn_mul: getFloatTensor<tf.Tensor1D>(19, [512]),
      bn_add: getFloatTensor<tf.Tensor1D>(20, [512])
    };

    const rgcn1: RGCNLayerWeights = {
      gather: getDequantizedTensor<tf.Tensor4D>(34, 35, [2, 4, 128, 128]),
      matmul: getDequantizedTensor<tf.Tensor2D>(38, 39, [512, 512]),
      add: getFloatTensor<tf.Tensor1D>(14, [512]),
      bn_mul: getFloatTensor<tf.Tensor1D>(21, [512]),
      bn_add: getFloatTensor<tf.Tensor1D>(22, [512])
    };

    const rgcn2: RGCNLayerWeights = {
      gather: getDequantizedTensor<tf.Tensor4D>(33, 34, [2, 4, 128, 128]),
      matmul: getDequantizedTensor<tf.Tensor2D>(39, 40, [512, 512]),
      add: getFloatTensor<tf.Tensor1D>(15, [512]),
      bn_mul: getFloatTensor<tf.Tensor1D>(23, [512]),
      bn_add: getFloatTensor<tf.Tensor1D>(24, [512])
    };

    const rgcn3: RGCNLayerWeights = {
      gather: getDequantizedTensor<tf.Tensor4D>(32, 33, [2, 4, 128, 128]),
      matmul: getDequantizedTensor<tf.Tensor2D>(40, 41, [512, 512]),
      add: getFloatTensor<tf.Tensor1D>(16, [512]),
      bn_mul: null,
      bn_add: null
    };

    const out_w = getDequantizedTensor<tf.Tensor2D>(31, 32, [4, 512]);
    const out_b = getFloatTensor<tf.Tensor1D>(31, [4]);

    return new GLNNModel({
      charEmb,
      bn0_mul,
      bn0_add,
      layers: [rgcn0, rgcn1, rgcn2, rgcn3],
      out_w,
      out_b
    });
  }

  predict(graph: GraphData): Int32Array {
    const numNodes = graph.nodeTypes.length;
    if (numNodes === 0) return new Int32Array(0);

    const numEdges = graph.edgeIndices.length;
    const srcNodes = graph.edgeIndices.map(e => e[0]);
    const dstNodes = graph.edgeIndices.map(e => e[1]);
    const edgeTypes = graph.edgeTypes;

    return tf.tidy(() => {
      const charIndicesTensor = tf.tensor1d(graph.charIndices, 'int32');
      const charToNodeTensor = tf.tensor1d(graph.charToNodeMap, 'int32');

      // 1. Character Embedding Lookup
      const charEmbs = tf.gather(this.weights.charEmb, charIndicesTensor);

      // 2. Initial Node Features (UnsortedSegmentMean)
      const nodeSums = tf.unsortedSegmentSum(charEmbs, charToNodeTensor, numNodes);
      const ones = tf.ones([graph.charToNodeMap.length, 1], 'float32');
      const nodeCounts = tf.maximum(tf.unsortedSegmentSum(ones, charToNodeTensor, numNodes), 1);
      const nodeInit = tf.div(nodeSums, nodeCounts);

      // 3. Initial BatchNorm
      const H0 = tf.relu(tf.add(tf.mul(nodeInit, this.weights.bn0_mul), this.weights.bn0_add));

      const srcTensor = tf.tensor1d(srcNodes, 'int32');
      const dstTensor = tf.tensor1d(dstNodes, 'int32');
      const edgeTypesTensor = tf.tensor1d(edgeTypes, 'int32');

      let H_curr = H0;
      let res = nodeInit;

      // 4. RGCN Convolution Layers
      for (const layer of this.weights.layers) {
        // Self-loop: H * W_self^T
        const H_self = tf.matMul(H_curr, layer.matmul, false, true);

        let H_msg = tf.zeros([numNodes, 512], 'float32');
        if (numEdges > 0) {
          const H_src = tf.gather(H_curr, srcTensor);
          const H_src_reshaped = tf.reshape(H_src, [numEdges, 4, 1, 128]);
          const W_rel = tf.gather(layer.gather, edgeTypesTensor);
          const M = tf.matMul(H_src_reshaped, W_rel);
          const M_flat = tf.reshape(M, [numEdges, 512]);

          const edgeKey: number[] = [];
          for (let e = 0; e < numEdges; e++) {
            edgeKey.push(dstNodes[e] * 2 + edgeTypes[e]);
          }
          const edgeKeyTensor = tf.tensor1d(edgeKey, 'int32');
          const edgeOnes = tf.ones([numEdges], 'float32');
          const inDeg = tf.unsortedSegmentSum(edgeOnes, edgeKeyTensor, numNodes * 2);
          const edgeDeg = tf.gather(inDeg, edgeKeyTensor);
          const edgeDegNorm = tf.expandDims(tf.maximum(edgeDeg, 1), 1);
          const M_norm = tf.div(M_flat, edgeDegNorm);

          H_msg = tf.unsortedSegmentSum(M_norm, dstTensor, numNodes);
        }

        const H_combined = tf.add(tf.add(tf.add(H_self, H_msg), layer.add), res);
        res = H_combined;

        if (layer.bn_mul && layer.bn_add) {
          H_curr = tf.relu(tf.add(tf.mul(H_combined, layer.bn_mul), layer.bn_add));
        } else {
          H_curr = H_combined;
        }
      }

      // 5. Output Dense Layer & ArgMax
      const logits = tf.add(tf.matMul(H_curr, this.weights.out_w, false, true), this.weights.out_b);
      const argmax = tf.argMax(logits, -1);
      return argmax.dataSync() as unknown as Int32Array;
    });
  }

  dispose(): void {
    tf.dispose([
      this.weights.charEmb,
      this.weights.bn0_mul,
      this.weights.bn0_add,
      this.weights.out_w,
      this.weights.out_b
    ]);
    for (const l of this.weights.layers) {
      tf.dispose([l.gather, l.matmul, l.add]);
      if (l.bn_mul) tf.dispose(l.bn_mul);
      if (l.bn_add) tf.dispose(l.bn_add);
    }
  }
}
