import { defineConfig } from 'vite';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  root: 'website',
  resolve: {
    alias: [
      {
        find: /^@tensorflow\/tfjs-tflite$/,
        replacement: path.resolve(__dirname, 'node_modules/@tensorflow/tfjs-tflite/dist/tf-tflite.es2017.js')
      },
      {
        find: /^\.\.\/tflite_web_api_client$/,
        replacement: path.resolve(__dirname, 'node_modules/@tensorflow/tfjs-tflite/dist/tflite_web_api_client.js')
      }
    ]
  },
  server: {
    port: 5173,
    open: false
  },
  build: {
    outDir: '../dist-website',
    emptyOutDir: true,
    target: 'es2022'
  }
});
