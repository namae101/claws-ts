import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs-core';
import { Segmenter, segKcc } from '../src/index';

type ViewMode = 'chips' | 'space' | 'zwsp' | 'json';

class AppController {
  private segmenter: Segmenter | null = null;
  private currentTokens: string[] = [];
  private currentMode: ViewMode = 'chips';
  private debounceTimer: number | null = null;
  private currentBackend: string = 'webgl';

  // DOM Elements
  private statusBadge = document.getElementById('status-badge') as HTMLElement;
  private statusText = document.getElementById('status-text') as HTMLElement;
  private metricsBadge = document.getElementById('metrics-badge') as HTMLElement;
  private textInput = document.getElementById('text-input') as HTMLTextAreaElement;
  private inputStats = document.getElementById('input-stats') as HTMLElement;
  private outputStats = document.getElementById('output-stats') as HTMLElement;
  private outputContainer = document.getElementById('output-container') as HTMLElement;
  private btnSegment = document.getElementById('btn-segment') as HTMLButtonElement;
  private autoSegmentCheckbox = document.getElementById('auto-segment') as HTMLInputElement;
  private btnCopy = document.getElementById('btn-copy') as HTMLButtonElement;
  private toast = document.getElementById('toast') as HTMLElement;
  private tabButtons = document.querySelectorAll<HTMLButtonElement>('.tab-btn');
  private presetButtons = document.querySelectorAll<HTMLButtonElement>('.btn-preset');
  private btnBackendWebgl = document.getElementById('btn-backend-webgl') as HTMLButtonElement;
  private btnBackendCpu = document.getElementById('btn-backend-cpu') as HTMLButtonElement;

  async init(): Promise<void> {
    this.bindEvents();
    this.updateInputStats();

    try {
      // Prefer WebGL GPU acceleration if available, fallback to CPU
      try {
        await tf.setBackend('webgl');
        await tf.ready();
      } catch (e) {
        console.warn('WebGL initialization failed, falling back to CPU:', e);
        await tf.setBackend('cpu');
        await tf.ready();
      }

      this.currentBackend = tf.getBackend();
      this.updateBackendUI();

      this.statusText.textContent = 'Loading GNN Model...';
      const startInit = performance.now();

      this.segmenter = await Segmenter.create({
        modelUrl: './models/model.tflite',
        vocabUrlOrData: './models/vocab.json'
      });

      const initTime = (performance.now() - startInit).toFixed(0);
      this.statusBadge.classList.add('ready');
      this.statusText.textContent = 'GNN Ready';
      this.btnSegment.disabled = false;

      const deviceLabel = this.currentBackend === 'webgl' ? '⚡ WebGL (GPU)' : '💻 CPU';
      this.metricsBadge.innerHTML = `Loaded in <span class="highlight">${initTime}ms</span> • Engine: <span class="device-tag ${this.currentBackend === 'cpu' ? 'cpu' : ''}">${deviceLabel}</span>`;

      // Initial segmentation
      await this.runSegmentation();
    } catch (err: unknown) {
      console.error('Failed to load Segmenter:', err);
      this.statusText.textContent = 'Error loading model';
      this.metricsBadge.textContent = err instanceof Error ? err.message : 'Unknown error';
    }
  }

  private bindEvents(): void {
    // Backend switcher buttons
    this.btnBackendWebgl?.addEventListener('click', async () => {
      await this.switchBackend('webgl');
    });

    this.btnBackendCpu?.addEventListener('click', async () => {
      await this.switchBackend('cpu');
    });

    // Input changes
    this.textInput.addEventListener('input', () => {
      this.updateInputStats();
      if (this.autoSegmentCheckbox.checked) {
        if (this.debounceTimer) {
          window.clearTimeout(this.debounceTimer);
        }
        this.debounceTimer = window.setTimeout(() => {
          this.runSegmentation();
        }, 120);
      }
    });

    // Keyboard shortcut (Ctrl+Enter / Cmd+Enter)
    this.textInput.addEventListener('keydown', (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        this.runSegmentation();
      }
    });

    // Segment Button
    this.btnSegment.addEventListener('click', () => {
      this.runSegmentation();
    });

    // View Mode Tabs
    this.tabButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const mode = btn.dataset.mode as ViewMode;
        if (mode) {
          this.setMode(mode);
        }
      });
    });

    // Preset Buttons
    this.presetButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        if (btn.dataset.clear === 'true') {
          this.textInput.value = '';
        } else if (btn.dataset.text) {
          this.textInput.value = btn.dataset.text;
        }
        this.updateInputStats();
        this.runSegmentation();
      });
    });

    // Copy Button
    this.btnCopy.addEventListener('click', () => {
      this.copyToClipboard();
    });
  }

  private async switchBackend(target: 'webgl' | 'cpu'): Promise<void> {
    if (this.currentBackend === target) return;

    try {
      this.statusText.textContent = `Switching to ${target.toUpperCase()}...`;
      await tf.setBackend(target);
      await tf.ready();
      this.currentBackend = tf.getBackend();
      this.updateBackendUI();
      this.statusText.textContent = 'GNN Ready';
      this.showToast(`Switched hardware engine to: ${target === 'webgl' ? 'GPU (WebGL)' : 'CPU'}`);
      await this.runSegmentation();
    } catch (err: unknown) {
      console.error(`Failed to switch backend to ${target}:`, err);
      this.showToast(`Error switching to ${target}`);
    }
  }

  private updateBackendUI(): void {
    const isWebgl = this.currentBackend === 'webgl';
    if (this.btnBackendWebgl) {
      this.btnBackendWebgl.classList.toggle('active', isWebgl);
    }
    if (this.btnBackendCpu) {
      this.btnBackendCpu.classList.toggle('active', !isWebgl);
    }
  }

  private updateInputStats(): void {
    const text = this.textInput.value;
    const charCount = text.length;
    const kccs = segKcc(text.trim());
    this.inputStats.textContent = `${charCount} chars • ${kccs.length} KCCs`;
  }

  private async runSegmentation(): Promise<void> {
    if (!this.segmenter) return;

    const text = this.textInput.value.trim();
    if (!text) {
      this.currentTokens = [];
      this.renderOutput();
      this.outputStats.textContent = '0 words';
      const isCpu = this.currentBackend === 'cpu';
      const deviceLabel = isCpu ? '💻 CPU' : '⚡ GPU (WebGL)';
      this.metricsBadge.innerHTML = `<span class="highlight">Latency:</span> 0 ms | 0 tokens <span class="device-tag ${isCpu ? 'cpu' : ''}">(${deviceLabel})</span>`;
      return;
    }

    const allKccs = segKcc(text);
    const maxChunkSize = 60;
    const totalChunks = Math.ceil(allKccs.length / maxChunkSize);

    // Set UI to loading state
    this.statusBadge.classList.add('processing');
    this.statusBadge.classList.remove('ready');
    this.statusText.textContent = totalChunks > 1 ? `Segmenting (0/${totalChunks})...` : 'Segmenting...';
    this.btnSegment.disabled = true;

    if (totalChunks > 1) {
      this.outputContainer.innerHTML = `
        <div class="loading-overlay">
          <div class="spinner"></div>
          <div id="loading-chunk-text">Processing large input in ${totalChunks} chunks...</div>
          <div class="progress-container">
            <div id="progress-fill" class="progress-fill"></div>
          </div>
        </div>
      `;
    }

    try {
      const startTime = performance.now();

      const tokens = await this.segmenter.segment(text, {
        maxKccsPerChunk: maxChunkSize,
        onProgress: (progress) => {
          this.statusText.textContent = `Segmenting (${progress.current}/${progress.total})...`;
          const fill = document.getElementById('progress-fill');
          if (fill) {
            fill.style.width = `${progress.percentage}%`;
          }
          const chunkText = document.getElementById('loading-chunk-text');
          if (chunkText) {
            chunkText.textContent = `Processing chunk ${progress.current} of ${progress.total} (${progress.percentage}%)...`;
          }
        }
      });

      const elapsed = (performance.now() - startTime).toFixed(1);
      this.currentTokens = tokens;

      const isCpu = this.currentBackend === 'cpu';
      const deviceLabel = isCpu ? '💻 CPU' : '⚡ GPU (WebGL)';
      const chunkInfo = totalChunks > 1 ? ` • ${totalChunks} chunks` : '';

      this.metricsBadge.innerHTML = `<span class="highlight">Latency:</span> ${elapsed} ms | ${tokens.length} tokens <span class="device-tag ${isCpu ? 'cpu' : ''}">(${deviceLabel}${chunkInfo})</span>`;
      this.outputStats.textContent = `${tokens.length} words`;
      this.renderOutput();
    } catch (err: unknown) {
      console.error('Segmentation error:', err);
      this.statusText.textContent = 'Error segmenting';
    } finally {
      this.statusBadge.classList.remove('processing');
      this.statusBadge.classList.add('ready');
      this.statusText.textContent = 'GNN Ready';
      this.btnSegment.disabled = false;
    }
  }
  private setMode(mode: ViewMode): void {
    this.currentMode = mode;
    this.tabButtons.forEach(btn => {
      btn.classList.toggle('active', btn.dataset.mode === mode);
    });
    this.renderOutput();
  }

  private renderOutput(): void {
    if (this.currentTokens.length === 0) {
      this.outputContainer.innerHTML = `
        <div style="color: var(--text-muted); text-align: center; margin-top: 5rem; font-size: 0.95rem;">
          No output to display. Enter Khmer text on the left panel to segment.
        </div>
      `;
      return;
    }

    switch (this.currentMode) {
      case 'chips': {
        const grid = document.createElement('div');
        grid.className = 'tokens-grid';
        this.currentTokens.forEach((token, index) => {
          const chip = document.createElement('span');
          chip.className = 'token-chip';
          chip.title = `Token #${index + 1}: ${token} (Click to copy)`;
          chip.innerHTML = `<span>${this.escapeHtml(token)}</span><span class="token-index">${index + 1}</span>`;
          chip.addEventListener('click', () => {
            navigator.clipboard.writeText(token);
            this.showToast(`Copied token: "${token}"`);
          });
          grid.appendChild(chip);
        });
        this.outputContainer.innerHTML = '';
        this.outputContainer.appendChild(grid);
        break;
      }
      case 'space': {
        const textElem = document.createElement('div');
        textElem.className = 'output-text';
        textElem.textContent = this.currentTokens.join(' ');
        this.outputContainer.innerHTML = '';
        this.outputContainer.appendChild(textElem);
        break;
      }
      case 'zwsp': {
        const textElem = document.createElement('div');
        textElem.className = 'output-text';
        textElem.textContent = this.currentTokens.join('\u200b');
        this.outputContainer.innerHTML = '';
        this.outputContainer.appendChild(textElem);
        break;
      }
      case 'json': {
        const codeElem = document.createElement('pre');
        codeElem.className = 'output-code';
        codeElem.textContent = JSON.stringify(this.currentTokens, null, 2);
        this.outputContainer.innerHTML = '';
        this.outputContainer.appendChild(codeElem);
        break;
      }
    }
  }

  private copyToClipboard(): void {
    if (this.currentTokens.length === 0) return;

    let contentToCopy = '';
    switch (this.currentMode) {
      case 'chips':
      case 'space':
        contentToCopy = this.currentTokens.join(' ');
        break;
      case 'zwsp':
        contentToCopy = this.currentTokens.join('\u200b');
        break;
      case 'json':
        contentToCopy = JSON.stringify(this.currentTokens, null, 2);
        break;
    }

    navigator.clipboard.writeText(contentToCopy);
    this.showToast('Copied output to clipboard!');
  }

  private showToast(message: string): void {
    this.toast.textContent = message;
    this.toast.classList.add('show');
    setTimeout(() => {
      this.toast.classList.remove('show');
    }, 2000);
  }

  private escapeHtml(str: string): string {
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }
}

// Bootstrap application
const app = new AppController();
app.init();
