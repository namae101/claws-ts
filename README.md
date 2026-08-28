# claws-ts

> **Disclaimer:** This project was developed mostly with AI, mixing and matching open-source tools and libraries for simple tasks. All NLP processing (segmentation, spell checking, Unicode normalization) runs 100% locally client-side in browser memory with zero remote server dependencies.

Fast and lightweight Khmer word segmentation using Graph Neural Networks (GNN) in TypeScript / WebAssembly for In-Browser & Node.js execution.

Port of [claws](https://github.com/Socret360/claws) (Python) to TypeScript running client-side with `@tensorflow/tfjs-tflite`.

---

## Features
- **100% In-Browser / Client-Side**: No backend server required.
- **Zero Privacy Risk**: Text segmentation happens locally in the browser memory.
- **Fast Execution**: Uses TFLite WebAssembly (Wasm) runtime for GNN model inference.

---

## Installation

```bash
npm install claws-ts @tensorflow/tfjs-core @tensorflow/tfjs-tflite
```

---

## Usage

### Browser (Vite / React / Vue / HTML)

```typescript
import { Segmenter } from 'claws-ts';

// Initialize Segmenter with model & vocab file URLs
const segmenter = await Segmenter.create({
  modelUrl: '/models/model.tflite',
  vocabUrlOrData: '/models/vocab.json',
  // Optional: Set custom CDN/static path for TFJS TFLite WASM binaries
  wasmPath: 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite/dist/'
});

// Segment Khmer Text
const text = "ខ្ញុំស្រឡាញ់ប្រទេសកម្ពុជា";
const tokens = await segmenter.segment(text);

console.log(tokens);
// Output: ["ខ្ញុំ", "ស្រឡាញ់", "ប្រទេស", "កម្ពុជា"]
```

---

## Static Assets Setup
Make sure to copy `model.tflite` and `vocab.json` from the `claws` repository into your public static asset directory (e.g. `./public/models/`).

---

## License
MIT
