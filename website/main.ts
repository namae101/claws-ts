import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs-core';
import {
  Segmenter,
  segKcc,
  chunkKccs,
  normalize,
  spellChecker,
  WordSpellCheckResult
} from '../src/index';

type ViewMode = 'chips' | 'space' | 'zwsp' | 'json';

interface TokenAnalysis {
  token: string;
  spellResult?: WordSpellCheckResult;
}

class AppController {
  private segmenter: Segmenter | null = null;
  private currentTokens: TokenAnalysis[] = [];
  private currentMode: ViewMode = 'chips';
  private debounceTimer: number | null = null;
  private currentBackend: string = 'webgl';
  private statusBadge = document.getElementById('status-badge') as HTMLElement;
  private statusText = document.getElementById('status-text') as HTMLElement;
  private metricsBadge = document.getElementById('metrics-badge') as HTMLElement;
  private textInput = document.getElementById('text-input') as HTMLTextAreaElement;
  private inputStats = document.getElementById('input-stats') as HTMLElement;
  private outputStats = document.getElementById('output-stats') as HTMLElement;
  private spellStats = document.getElementById('spell-stats') as HTMLElement;
  private outputContainer = document.getElementById('output-container') as HTMLElement;
  private btnSegment = document.getElementById('btn-segment') as HTMLButtonElement;
  private autoSegmentCheckbox = document.getElementById('auto-segment') as HTMLInputElement;
  private toggleNormalize = document.getElementById('toggle-normalize') as HTMLInputElement;
  private toggleSpellcheck = document.getElementById('toggle-spellcheck') as HTMLInputElement;
  private btnCopy = document.getElementById('btn-copy') as HTMLButtonElement;
  private toast = document.getElementById('toast') as HTMLElement;
  private suggestionPopover = document.getElementById('suggestion-popover') as HTMLElement;
  private tabButtons = document.querySelectorAll<HTMLButtonElement>('.tab-btn');
  private presetButtons = document.querySelectorAll<HTMLButtonElement>('.btn-preset');
  private btnBackendWebgl = document.getElementById('btn-backend-webgl') as HTMLButtonElement;
  private btnBackendCpu = document.getElementById('btn-backend-cpu') as HTMLButtonElement;
  private panelInput = document.getElementById('panel-input') as HTMLElement;
  private panelOutput = document.getElementById('panel-output') as HTMLElement;
  private tabInputBtn = document.getElementById('tab-input-btn') as HTMLButtonElement;
  private tabOutputBtn = document.getElementById('tab-output-btn') as HTMLButtonElement;
  private mobileOutputBadge = document.getElementById('mobile-output-badge') as HTMLElement;
  private btnAck = document.getElementById('btn-ack') as HTMLButtonElement;
  private ackDialog = document.getElementById('ack-dialog') as HTMLDialogElement;
  private btnCloseDialog = document.getElementById('btn-close-dialog') as HTMLButtonElement;
  private btnDialogDone = document.getElementById('btn-dialog-done') as HTMLButtonElement;
  private btnTheme = document.getElementById('btn-theme') as HTMLButtonElement;
  private currentTheme: 'light' | 'dark' = 'light';

  async init(): Promise<void> {
    this.initTheme();
    this.bindEvents();
    this.updateInputStats();
    try {
      // 1. Initialize WebGL / CPU backend
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

      this.statusText.textContent = 'Loading GNN & Dictionary...';
      const startInit = performance.now();

      // 2. Initialize Segmenter & SpellChecker in parallel
      const [segInstance] = await Promise.all([
        Segmenter.create({
          modelUrl: './models/model.tflite',
          vocabUrlOrData: './models/vocab.json'
        }),
        spellChecker.init()
      ]);

      this.segmenter = segInstance;

      const initTime = (performance.now() - startInit).toFixed(0);
      this.statusBadge.classList.add('ready');
      this.statusText.textContent = 'GNN Ready';
      this.btnSegment.disabled = false;

      const deviceLabel = this.currentBackend === 'webgl' ? '⚡ WebGL (GPU)' : '💻 CPU';
      const dictCount = spellChecker.getWordCount();
      this.metricsBadge.innerHTML = `Loaded in <span class="highlight">${initTime}ms</span> • <span class="device-tag ${this.currentBackend === 'cpu' ? 'cpu' : ''}">${deviceLabel}</span> • Dictionary: <span class="highlight">${dictCount.toLocaleString()}</span> words`;

      // Initial segmentation (only if auto-segment is enabled)
      if (this.autoSegmentCheckbox?.checked) {
        await this.runSegmentation();
      }
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

    this.autoSegmentCheckbox?.addEventListener('change', () => {
      if (this.autoSegmentCheckbox.checked) {
        this.runSegmentation();
      }
    });

    this.toggleNormalize?.addEventListener('change', () => {
      if (this.currentTokens.length > 0 || this.autoSegmentCheckbox?.checked) {
        this.runSegmentation();
      }
    });

    this.toggleSpellcheck?.addEventListener('change', () => {
      if (this.currentTokens.length > 0 || this.autoSegmentCheckbox?.checked) {
        this.runSegmentation();
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
          this.updateInputStats();
          this.runSegmentation();
        } else if (btn.dataset.text) {
          this.textInput.value = btn.dataset.text;
          this.updateInputStats();
          if (this.autoSegmentCheckbox?.checked) {
            this.runSegmentation();
          }
        }
      });
    });

    // Mobile Bottom Tab Navigation
    this.tabInputBtn?.addEventListener('click', () => {
      this.switchMobileTab('input');
    });

    this.tabOutputBtn?.addEventListener('click', () => {
      this.switchMobileTab('output');
    });
    // Copy Button
    this.btnCopy.addEventListener('click', () => {
      this.copyToClipboard();
    });

    // Close popover when clicking outside
    document.addEventListener('click', (e: MouseEvent) => {
      if (!this.suggestionPopover.contains(e.target as Node) && !(e.target as HTMLElement).closest('.token-chip.misspelled')) {
        this.hidePopover();
      }
    });
    // Prevent output scroll when suggestion popover is open
    this.outputContainer.addEventListener('wheel', this.wheelBlocker, { passive: false });

    // Theme Toggle
    this.btnTheme?.addEventListener('click', () => {
      this.toggleTheme();
    });

    // Acknowledgments Dialog
    this.btnAck?.addEventListener('click', () => {
      this.ackDialog?.showModal();
    });

    this.btnCloseDialog?.addEventListener('click', () => {
      this.ackDialog?.close();
    });

    this.btnDialogDone?.addEventListener('click', () => {
      this.ackDialog?.close();
    });

    // Close dialog if backdrop is clicked
    this.ackDialog?.addEventListener('click', (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (target === this.ackDialog) {
        this.ackDialog.close();
      }
    });
  }

  private initTheme(): void {
    const saved = localStorage.getItem('claws_theme') as 'light' | 'dark' | null;
    this.setTheme(saved || 'light');
  }

  private setTheme(theme: 'light' | 'dark'): void {
    this.currentTheme = theme;
    if (theme === 'dark') {
      document.documentElement.setAttribute('data-theme', 'dark');
    } else {
      document.documentElement.removeAttribute('data-theme');
    }
    localStorage.setItem('claws_theme', theme);
  }

  private toggleTheme(): void {
    const next = this.currentTheme === 'dark' ? 'light' : 'dark';
    this.setTheme(next);
    this.showToast(`Switched to ${next} mode`);
  }

  private switchMobileTab(tab: 'input' | 'output'): void {
    if (tab === 'input') {
      this.panelInput?.classList.add('active-panel');
      this.panelOutput?.classList.remove('active-panel');
      this.tabInputBtn?.classList.add('active');
      this.tabOutputBtn?.classList.remove('active');
    } else {
      this.panelOutput?.classList.add('active-panel');
      this.panelInput?.classList.remove('active-panel');
      this.tabOutputBtn?.classList.add('active');
      this.tabInputBtn?.classList.remove('active');
    }
    this.hidePopover();
  }

  private wheelBlocker = (e: WheelEvent): void => {
    if (this.suggestionPopover && this.suggestionPopover.classList.contains('show')) {
      e.preventDefault();
    }
  };

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
      if (this.currentTokens.length > 0 || this.autoSegmentCheckbox?.checked) {
        await this.runSegmentation();
      }
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

    let text = this.textInput.value.trim();
    if (!text) {
      this.currentTokens = [];
      this.renderOutput();
      this.outputStats.textContent = '0 words';
      this.spellStats.style.display = 'none';
      const isCpu = this.currentBackend === 'cpu';
      const deviceLabel = isCpu ? '💻 CPU' : '⚡ GPU (WebGL)';
      this.metricsBadge.innerHTML = `<span class="highlight">Latency:</span> 0 ms | 0 tokens <span class="device-tag ${isCpu ? 'cpu' : ''}">(${deviceLabel})</span>`;
      return;
    }

    // Optional Unicode Normalization
    if (this.toggleNormalize && this.toggleNormalize.checked) {
      text = normalize(text);
    }

    const allKccs = segKcc(text);
    const maxChunkSize = 60;
    const totalChunks = chunkKccs(allKccs, maxChunkSize).length;

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

      const rawTokens = await this.segmenter.segment(text, {
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

      // Perform Spell Checking if enabled
      const doSpellcheck = this.toggleSpellcheck ? this.toggleSpellcheck.checked : true;
      let typoCount = 0;

      this.currentTokens = rawTokens.map(tok => {
        if (!doSpellcheck) {
          return { token: tok };
        }
        const spellResult = spellChecker.checkWord(tok, 6, 5);
        if (!spellResult.is_correct) {
          typoCount++;
        }
        return {
          token: tok,
          spellResult
        };
      });

      // Update Spell Stats badge
      if (doSpellcheck && typoCount > 0) {
        this.spellStats.style.display = 'inline-block';
        this.spellStats.textContent = `${typoCount} typo${typoCount > 1 ? 's' : ''}`;
      } else {
        this.spellStats.style.display = 'none';
      }

      const isCpu = this.currentBackend === 'cpu';
      const deviceLabel = isCpu ? '💻 CPU' : '⚡ GPU (WebGL)';
      const chunkInfo = totalChunks > 1 ? ` • ${totalChunks} chunks` : '';
      this.outputStats.textContent = `${rawTokens.length} words`;
      if (this.mobileOutputBadge) {
        this.mobileOutputBadge.style.display = rawTokens.length > 0 ? 'inline-block' : 'none';
        this.mobileOutputBadge.textContent = String(rawTokens.length);
      }
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
    this.hidePopover();
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
        this.currentTokens.forEach((item, index) => {
          const chip = document.createElement('span');
          const isMisspelled = item.spellResult && !item.spellResult.is_correct;

          chip.className = `token-chip ${isMisspelled ? 'misspelled' : ''}`;
          chip.title = isMisspelled
            ? `⚠️ Possible spelling mistake: "${item.token}". Click for suggestions!`
            : `Token #${index + 1}: ${item.token} (Click to copy)`;

          chip.innerHTML = `
            <span>${this.escapeHtml(item.token)}</span>
            <span class="token-index">${index + 1}</span>
            ${isMisspelled ? '<span class="spell-dot"></span>' : ''}
          `;

          chip.addEventListener('click', (e: MouseEvent) => {
            if (isMisspelled && item.spellResult) {
              e.stopPropagation();
              this.showPopover(chip, item.token, item.spellResult.suggestions || []);
            } else {
              navigator.clipboard.writeText(item.token);
              this.showToast(`Copied token: "${item.token}"`);
            }
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
        textElem.textContent = this.currentTokens.map(t => t.token).join(' ');
        this.outputContainer.innerHTML = '';
        this.outputContainer.appendChild(textElem);
        break;
      }
      case 'zwsp': {
        const textElem = document.createElement('div');
        textElem.className = 'output-text';
        textElem.textContent = this.currentTokens.map(t => t.token).join('\u200b');
        this.outputContainer.innerHTML = '';
        this.outputContainer.appendChild(textElem);
        break;
      }
      case 'json': {
        const codeElem = document.createElement('pre');
        codeElem.className = 'output-code';
        const formatted = this.currentTokens.map(t => ({
          word: t.token,
          is_correct: t.spellResult ? t.spellResult.is_correct : true,
          suggestions: t.spellResult ? t.spellResult.suggestions : []
        }));
        codeElem.textContent = JSON.stringify(formatted, null, 2);
        this.outputContainer.innerHTML = '';
        this.outputContainer.appendChild(codeElem);
        break;
      }
    }
  }

  private showPopover(anchor: HTMLElement, originalWord: string, suggestions: string[]): void {
    const rect = anchor.getBoundingClientRect();
    const hasSuggestions = suggestions && suggestions.length > 0;

    const listHtml = hasSuggestions
      ? suggestions.slice(0, 6).map(s => `
          <div class="suggestion-item" data-suggestion="${this.escapeHtml(s)}">
            <span>${this.escapeHtml(s)}</span>
            <span class="suggestion-type">Replace</span>
          </div>
        `).join('')
      : `<div class="no-suggestions">No suggestions available</div>`;

    this.suggestionPopover.innerHTML = `
      <div class="suggestion-header">
        <span>Suggestions for "${this.escapeHtml(originalWord)}"</span>
      </div>
      <div class="suggestion-list">
        ${listHtml}
      </div>
    `;

    this.suggestionPopover.classList.add('show');
    this.outputContainer.classList.add('scroll-locked');
    const popoverWidth = Math.min(300, window.innerWidth - 24);
    const popoverHeight = 220;
    let left = Math.max(12, Math.min(window.innerWidth - popoverWidth - 12, rect.left));
    let top = rect.bottom + 6;
    if (top + popoverHeight > window.innerHeight - 60) {
      top = Math.max(12, rect.top - popoverHeight - 6);
    }
    this.suggestionPopover.style.top = `${top}px`;
    this.suggestionPopover.style.left = `${left}px`;
    const items = this.suggestionPopover.querySelectorAll<HTMLElement>('.suggestion-item');
    items.forEach(el => {
      el.addEventListener('click', () => {
        const replacement = el.dataset.suggestion;
        if (replacement) {
          this.replaceWord(originalWord, replacement);
        }
      });
    });
  }

  private hidePopover(): void {
    this.suggestionPopover.classList.remove('show');
    this.outputContainer.classList.remove('scroll-locked');
  }
  private replaceWord(originalWord: string, replacement: string): void {
    const currentText = this.textInput.value;
    const updated = currentText.replace(originalWord, replacement);
    this.textInput.value = updated;
    this.hidePopover();
    this.updateInputStats();
    this.showToast(`Replaced "${originalWord}" with "${replacement}"`);
    this.runSegmentation();
  }

  private copyToClipboard(): void {
    if (this.currentTokens.length === 0) return;

    let contentToCopy = '';
    switch (this.currentMode) {
      case 'chips':
      case 'space':
        contentToCopy = this.currentTokens.map(t => t.token).join(' ');
        break;
      case 'zwsp':
        contentToCopy = this.currentTokens.map(t => t.token).join('\u200b');
        break;
      case 'json':
        contentToCopy = JSON.stringify(this.currentTokens.map(t => t.token), null, 2);
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
