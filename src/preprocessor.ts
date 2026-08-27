export interface GraphData {
  charIndices: number[];
  charToNodeMap: number[];
  nodeTypes: number[];
  edgeIndices: number[][];
  edgeTypes: number[];
}

export class TextGraphEncoder {
  private vocabMap: Map<string, number>;
  private unkIndex: number;
  private directed: boolean;

  constructor(
    vocab: string[],
    directed: boolean = true
  ) {
    this.vocabMap = new Map();
    vocab.forEach((v, idx) => this.vocabMap.set(v, idx));
    this.unkIndex = this.vocabMap.get('<UNK/>') ?? 0;
    this.directed = directed;
  }

  public get vocabSize(): number {
    return this.vocabMap.size;
  }
  public encode(kccs: string[], nodeOffset: number = 0): GraphData {
    const numKccs = kccs.length;
    const eKccPrev: number[][] = [];
    const eKccNext: number[][] = [];

    if (numKccs > 1) {
      for (let i = 0; i < numKccs - 1; i++) {
        eKccPrev.push([i + nodeOffset, i + 1 + nodeOffset]);
        eKccNext.push([i + 1 + nodeOffset, i + nodeOffset]);
      }
    }

    const nodeTypes = new Array(numKccs).fill(0);
    const edgeIndices = [...eKccNext, ...eKccPrev];

    let edgeTypes: number[];
    if (this.directed) {
      edgeTypes = [
        ...new Array(eKccNext.length).fill(0),
        ...new Array(eKccPrev.length).fill(1)
      ];
    } else {
      edgeTypes = new Array(edgeIndices.length).fill(0);
    }

    const chars = kccs.join('').split('');
    const charIndices = chars.map(c => this.vocabMap.get(c) ?? this.unkIndex);

    const charToNodeMap: number[] = [];
    kccs.forEach((kcc, idx) => {
      for (let c = 0; c < kcc.length; c++) {
        charToNodeMap.push(idx + nodeOffset);
      }
    });

    return {
      charIndices,
      charToNodeMap,
      nodeTypes,
      edgeIndices: edgeIndices.length > 0 ? edgeIndices : [],
      edgeTypes
    };
  }
}
