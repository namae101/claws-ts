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
  tokenIndex: number;
  startIndex: number;
  endIndex: number;
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
  private btnDict = document.getElementById('btn-dict') as HTMLButtonElement;
  private dictDialog = document.getElementById('dict-dialog') as HTMLDialogElement;
  private btnCloseDictDialog = document.getElementById('btn-close-dict-dialog') as HTMLButtonElement;
  private btnDictDone = document.getElementById('btn-dialog-done') as HTMLButtonElement;
  private dictSearchInput = document.getElementById('dict-search-input') as HTMLInputElement;
  private btnClearDictSearch = document.getElementById('btn-clear-dict-search') as HTMLButtonElement;
  private dictFilterPills = document.querySelectorAll<HTMLButtonElement>('.dict-filter-pill');
  private dictSortSelect = document.getElementById('dict-sort-select') as HTMLSelectElement;
  private dictStatsText = document.getElementById('dict-stats-text') as HTMLElement;
  private dictWordsGrid = document.getElementById('dict-words-grid') as HTMLElement;
  private dictLoadMore = document.getElementById('dict-load-more') as HTMLElement;
  private btnDictLoadMore = document.getElementById('btn-dict-load-more') as HTMLButtonElement;
  private btnCustomWords = document.getElementById('btn-custom-words') as HTMLButtonElement;
  private customWordsCount = document.getElementById('custom-words-count') as HTMLElement;
  private customWordsDialog = document.getElementById('custom-words-dialog') as HTMLDialogElement;
  private btnCloseCustomDialog = document.getElementById('btn-close-custom-dialog') as HTMLButtonElement;
  private btnCustomDialogDone = document.getElementById('btn-custom-dialog-done') as HTMLButtonElement;
  private customAddInput = document.getElementById('custom-add-input') as HTMLInputElement;
  private btnCustomAdd = document.getElementById('btn-custom-add') as HTMLButtonElement;
  private customImportFile = document.getElementById('custom-import-file') as HTMLInputElement;
  private btnCustomImport = document.getElementById('btn-custom-import') as HTMLButtonElement;
  private btnCustomExport = document.getElementById('btn-custom-export') as HTMLButtonElement;
  private btnCustomClearAll = document.getElementById('btn-custom-clear-all') as HTMLButtonElement;
  private customStatsText = document.getElementById('custom-stats-text') as HTMLElement;
  private customWordsGrid = document.getElementById('custom-words-grid') as HTMLElement;
  private customWordsList: string[] = [];

  private dictCurrentMode: 'contains' | 'fuzzy' | 'exact' = 'contains';
  private dictCurrentOffset: number = 0;
  private dictPageSize: number = 80;
  private dictSearchDebounce: number | null = null;
  private btnTheme = document.getElementById('btn-theme') as HTMLButtonElement;
  private currentTheme: 'light' | 'dark' = 'light';
  private sessionIgnoredWords: Set<string> = new Set();

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

      this.loadCustomDictionary();
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

    // Dictionary Explorer
    this.btnDict?.addEventListener('click', () => {
      this.openDictDialog();
    });

    this.btnCloseDictDialog?.addEventListener('click', () => {
      this.dictDialog?.close();
    });

    const btnDictDone = document.getElementById('btn-dict-done');
    btnDictDone?.addEventListener('click', () => {
      this.dictDialog?.close();
    });

    this.dictDialog?.addEventListener('click', (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (target === this.dictDialog) {
        this.dictDialog.close();
      }
    });

    this.dictSearchInput?.addEventListener('input', () => {
      const val = this.dictSearchInput.value;
      if (this.btnClearDictSearch) {
        this.btnClearDictSearch.style.display = val ? 'block' : 'none';
      }
      if (this.dictSearchDebounce) {
        window.clearTimeout(this.dictSearchDebounce);
      }
      this.dictSearchDebounce = window.setTimeout(() => {
        this.renderDictWords(true);
      }, 100);
    });

    this.btnClearDictSearch?.addEventListener('click', () => {
      this.dictSearchInput.value = '';
      this.btnClearDictSearch.style.display = 'none';
      this.dictSearchInput.focus();
      this.renderDictWords(true);
    });

    this.dictFilterPills.forEach(pill => {
      pill.addEventListener('click', () => {
        this.dictFilterPills.forEach(p => p.classList.remove('active'));
        pill.classList.add('active');
        const mode = pill.dataset.dictMode;
        if (mode === 'fuzzy' || mode === 'exact' || mode === 'contains') {
          this.dictCurrentMode = mode;
        } else {
          this.dictCurrentMode = 'contains';
        }
        this.renderDictWords(true);
      });
    });

    this.dictSortSelect?.addEventListener('change', () => {
      this.renderDictWords(true);
    });

    this.btnDictLoadMore?.addEventListener('click', () => {
      this.renderDictWords(false);
    });

    // Custom Words Manager
    this.btnCustomWords?.addEventListener('click', () => {
      this.openCustomWordsDialog();
    });

    this.btnCloseCustomDialog?.addEventListener('click', () => {
      this.customWordsDialog?.close();
    });

    this.btnCustomDialogDone?.addEventListener('click', () => {
      this.customWordsDialog?.close();
    });

    this.customWordsDialog?.addEventListener('click', (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (target === this.customWordsDialog) {
        this.customWordsDialog.close();
      }
    });

    this.btnCustomAdd?.addEventListener('click', () => {
      this.handleAddCustomWord();
    });

    this.customAddInput?.addEventListener('keydown', (e: KeyboardEvent) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        this.handleAddCustomWord();
      }
    });

    this.btnCustomExport?.addEventListener('click', () => {
      this.exportCustomWords();
    });

    this.btnCustomImport?.addEventListener('click', () => {
      this.customImportFile?.click();
    });

    this.customImportFile?.addEventListener('change', (e: Event) => {
      const input = e.target as HTMLInputElement;
      const file = input.files?.[0];
      if (file) {
        this.importCustomWords(file);
      }
      input.value = '';
    });

    this.btnCustomClearAll?.addEventListener('click', () => {
      this.clearAllCustomWords();
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

  private openDictDialog(): void {
    this.dictDialog?.showModal();
    this.dictSearchInput?.focus();
    this.renderDictWords(true);
  }

  private renderDictWords(reset: boolean = false): void {
    if (!this.dictWordsGrid) return;
    if (reset) {
      this.dictCurrentOffset = 0;
      this.dictWordsGrid.innerHTML = '';
    }

    const query = this.dictSearchInput ? this.dictSearchInput.value.trim() : '';
    const sortVal = this.dictSortSelect ? this.dictSortSelect.value : 'freq-desc';
    const [sortBy, sortOrder] = sortVal.split('-') as ['freq' | 'alpha' | 'length', 'asc' | 'desc'];

    const result = spellChecker.filterWords({
      query,
      mode: this.dictCurrentMode,
      sortBy: sortBy || 'freq',
      sortOrder: sortOrder || 'desc',
      limit: this.dictPageSize,
      offset: this.dictCurrentOffset
    });

    const totalCount = spellChecker.getWordCount();
    const isSearching = query.length > 0;
    
    if (this.dictStatsText) {
      if (isSearching) {
        this.dictStatsText.textContent = `Found ${result.total.toLocaleString()} matching words out of ${totalCount.toLocaleString()}`;
      } else {
        this.dictStatsText.textContent = `Total ${totalCount.toLocaleString()} vocabulary words in dictionary`;
      }
    }

    if (result.total === 0) {
      this.dictWordsGrid.innerHTML = `<div class="dict-empty-state">No matching dictionary words found for "${this.escapeHtml(query)}"</div>`;
      if (this.dictLoadMore) this.dictLoadMore.style.display = 'none';
      return;
    }

    const fragment = document.createDocumentFragment();
    result.words.forEach(item => {
      const card = document.createElement('div');
      card.className = 'dict-word-card';
      card.title = `Click to append "${item.word}" to input`;
      const freqLabel = item.freq > 1 ? item.freq.toLocaleString() : `${item.word.length} chars`;
      card.innerHTML = `
        <span class="dict-word-text">${this.escapeHtml(item.word)}</span>
        <span class="dict-word-freq">${freqLabel}</span>
      `;
      card.addEventListener('click', () => {
        this.insertWordIntoInput(item.word);
      });
      fragment.appendChild(card);
    });

    this.dictWordsGrid.appendChild(fragment);
    this.dictCurrentOffset += result.words.length;

    if (this.dictLoadMore) {
      if (this.dictCurrentOffset < result.total) {
        this.dictLoadMore.style.display = 'flex';
      } else {
        this.dictLoadMore.style.display = 'none';
      }
    }
  }

  private insertWordIntoInput(word: string): void {
    const cur = this.textInput.value.trim();
    if (!cur) {
      this.textInput.value = word;
    } else {
      this.textInput.value = cur + ' ' + word;
    }
    this.updateInputStats();
    this.showToast(`Added "${word}" to input`);
    if (this.autoSegmentCheckbox?.checked) {
      this.runSegmentation();
    }
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

      const rawInputText = this.textInput.value;
      let searchCursor = 0;

      this.currentTokens = rawTokens.map((tok, tokenIndex) => {
        let start = -1;
        let end = -1;
        const foundPos = rawInputText.indexOf(tok, searchCursor);
        if (foundPos !== -1) {
          start = foundPos;
          end = foundPos + tok.length;
          searchCursor = end;
        }
        if (!doSpellcheck || this.sessionIgnoredWords.has(tok)) {
          return {
            token: tok,
            tokenIndex,
            startIndex: start,
            endIndex: end,
            spellResult: { word: tok, normalized_word: tok, is_correct: true, suggestions: [] }
          };
        }
        const spellResult = spellChecker.checkWord(tok, 6, 5);
        if (spellResult && !spellResult.is_correct) {
          typoCount++;
        }
        return {
          token: tok,
          tokenIndex,
          startIndex: start,
          endIndex: end,
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
              this.showPopover(chip, item, item.spellResult.suggestions || []);
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

  private showPopover(anchor: HTMLElement, tokenItem: TokenAnalysis, suggestions: string[]): void {
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
        <span>Suggestions for "${this.escapeHtml(tokenItem.token)}"</span>
      </div>
      <div class="suggestion-list">
        ${listHtml}
      </div>
      <div class="popover-actions">
        <button class="btn-popover-action btn-popover-ignore" data-action="ignore">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle></svg>
          <span>Ignore for this session</span>
        </button>
        <button class="btn-popover-action btn-popover-add" data-action="add">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="16"></line><line x1="8" y1="12" x2="16" y2="12"></line></svg>
          <span>Add to custom dictionary</span>
        </button>
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
          this.replaceWord(tokenItem, replacement);
        }
      });
    });

    const btnIgnore = this.suggestionPopover.querySelector<HTMLElement>('.btn-popover-ignore');
    btnIgnore?.addEventListener('click', (e) => {
      e.stopPropagation();
      this.ignoreWordForSession(tokenItem.token);
    });

    const btnAdd = this.suggestionPopover.querySelector<HTMLElement>('.btn-popover-add');
    btnAdd?.addEventListener('click', (e) => {
      e.stopPropagation();
      this.addWordToCustomDictionary(tokenItem.token);
    });
  }

  private hidePopover(): void {
    this.suggestionPopover.classList.remove('show');
    this.outputContainer.classList.remove('scroll-locked');
  }
  private replaceWord(tokenItem: TokenAnalysis, replacement: string): void {
    const currentText = this.textInput.value;
    const oldToken = tokenItem.token;
    const lengthDiff = replacement.length - oldToken.length;

    // 1. Update textInput value at exact position
    if (tokenItem.startIndex >= 0 && tokenItem.endIndex > tokenItem.startIndex && currentText.slice(tokenItem.startIndex, tokenItem.endIndex) === oldToken) {
      const before = currentText.slice(0, tokenItem.startIndex);
      const after = currentText.slice(tokenItem.endIndex);
      this.textInput.value = before + replacement + after;
    } else {
      let targetOccurrence = 0;
      for (let i = 0; i < tokenItem.tokenIndex; i++) {
        if (this.currentTokens[i]?.token === oldToken) {
          targetOccurrence++;
        }
      }
      let pos = -1;
      for (let count = 0; count <= targetOccurrence; count++) {
        pos = currentText.indexOf(oldToken, pos + 1);
        if (pos === -1) break;
      }
      if (pos !== -1) {
        const before = currentText.slice(0, pos);
        const after = currentText.slice(pos + oldToken.length);
        this.textInput.value = before + replacement + after;
        tokenItem.startIndex = pos;
        tokenItem.endIndex = pos + oldToken.length;
      } else {
        this.textInput.value = currentText.replace(oldToken, replacement);
      }
    }

    // 2. In-place update of currentTokens
    tokenItem.token = replacement;
    tokenItem.endIndex = tokenItem.startIndex + replacement.length;
    tokenItem.spellResult = spellChecker.checkWord(replacement, 6, 5);

    // Adjust start and end index for all subsequent tokens
    if (lengthDiff !== 0) {
      for (let i = tokenItem.tokenIndex + 1; i < this.currentTokens.length; i++) {
        const nextToken = this.currentTokens[i];
        if (nextToken && nextToken.startIndex >= 0) {
          nextToken.startIndex += lengthDiff;
          nextToken.endIndex += lengthDiff;
        }
      }
    }

    // 3. Update UI states & stats
    this.hidePopover();
    this.updateInputStats();

    // 4. Update spell stats badge
    let typoCount = 0;
    this.currentTokens.forEach(t => {
      if (t.spellResult && !t.spellResult.is_correct) {
        typoCount++;
      }
    });

    if (this.toggleSpellcheck?.checked && typoCount > 0) {
      this.spellStats.style.display = 'inline-block';
      this.spellStats.textContent = `${typoCount} typo${typoCount > 1 ? 's' : ''}`;
    } else {
      this.spellStats.style.display = 'none';
    }

    // 5. In-place re-render without neural network re-segmentation
    this.renderOutput();
    this.showToast(`Replaced "${oldToken}" with "${replacement}"`);
  }

  private loadCustomDictionary(): void {
    try {
      const saved = localStorage.getItem('claws_custom_words');
      if (saved) {
        const words = JSON.parse(saved) as string[];
        if (Array.isArray(words)) {
          this.customWordsList = words;
          spellChecker.addWords(words);
          this.updateCustomWordsCountBadge();
        }
      }
    } catch (e) {
      console.warn('Failed to load custom dictionary:', e);
    }
  }

  private updateCustomWordsCountBadge(): void {
    if (this.customWordsCount) {
      const count = this.customWordsList.length;
      this.customWordsCount.style.display = count > 0 ? 'inline-block' : 'none';
      this.customWordsCount.textContent = String(count);
    }
    if (this.customStatsText) {
      const count = this.customWordsList.length;
      this.customStatsText.textContent = `${count} custom word${count === 1 ? '' : 's'} saved in localStorage`;
    }
  }

  private openCustomWordsDialog(): void {
    this.customWordsDialog?.showModal();
    this.customAddInput?.focus();
    this.renderCustomWordsList();
  }

  private renderCustomWordsList(): void {
    if (!this.customWordsGrid) return;
    this.updateCustomWordsCountBadge();

    if (this.customWordsList.length === 0) {
      this.customWordsGrid.innerHTML = `
        <div class="custom-empty-state">
          <span>No custom words added yet.</span>
          <span style="font-size: 0.8rem; color: var(--text-muted);">Add new words above or click "Add to custom dictionary" in spell check suggestions.</span>
        </div>
      `;
      return;
    }

    this.customWordsGrid.innerHTML = '';
    const fragment = document.createDocumentFragment();

    this.customWordsList.forEach((word) => {
      const card = document.createElement('div');
      card.className = 'custom-word-card';
      card.innerHTML = `
        <span class="custom-word-text">${this.escapeHtml(word)}</span>
        <button class="btn-remove-word" title="Remove word">✕</button>
      `;
      const btnRemove = card.querySelector('.btn-remove-word');
      btnRemove?.addEventListener('click', (e) => {
        e.stopPropagation();
        this.removeCustomWord(word);
      });
      fragment.appendChild(card);
    });

    this.customWordsGrid.appendChild(fragment);
  }

  private handleAddCustomWord(): void {
    const word = this.customAddInput?.value.trim();
    if (!word) return;
    this.addWordToCustomDictionary(word);
    if (this.customAddInput) {
      this.customAddInput.value = '';
      this.customAddInput.focus();
    }
    this.renderCustomWordsList();
  }

  private addWordToCustomDictionary(word: string): void {
    const norm = normalize(word.trim());
    if (!norm) return;

    spellChecker.addWord(norm);
    this.sessionIgnoredWords.add(norm);

    if (!this.customWordsList.includes(norm)) {
      this.customWordsList.push(norm);
      try {
        localStorage.setItem('claws_custom_words', JSON.stringify(this.customWordsList));
      } catch (e) {
        console.warn('Failed to save to localStorage:', e);
      }
    }

    this.updateCustomWordsCountBadge();
    this.markWordAsCorrectInSession(norm);
    this.hidePopover();
    this.showToast(`Added "${norm}" to custom dictionary`);
  }

  private removeCustomWord(word: string): void {
    this.customWordsList = this.customWordsList.filter(w => w !== word);
    try {
      localStorage.setItem('claws_custom_words', JSON.stringify(this.customWordsList));
    } catch (e) {
      console.warn('Failed to save to localStorage:', e);
    }
    this.updateCustomWordsCountBadge();
    this.renderCustomWordsList();
    this.showToast(`Removed "${word}" from custom dictionary`);
  }

  private clearAllCustomWords(): void {
    if (this.customWordsList.length === 0) return;
    if (!confirm('Are you sure you want to clear all custom words?')) return;
    this.customWordsList = [];
    try {
      localStorage.removeItem('claws_custom_words');
    } catch (e) {
      console.warn('Failed to clear localStorage:', e);
    }
    this.updateCustomWordsCountBadge();
    this.renderCustomWordsList();
    this.showToast('Cleared all custom words');
  }

  private exportCustomWords(): void {
    if (this.customWordsList.length === 0) {
      this.showToast('No custom words to export');
      return;
    }
    const content = this.customWordsList.join('\n');
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `khmer_custom_words_${new Date().toISOString().slice(0, 10)}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    this.showToast(`Exported ${this.customWordsList.length} custom words`);
  }

  private async importCustomWords(file: File): Promise<void> {
    try {
      const text = await file.text();
      let importedWords: string[] = [];

      if (file.name.endsWith('.json')) {
        try {
          const parsed = JSON.parse(text);
          if (Array.isArray(parsed)) {
            importedWords = parsed.map(w => typeof w === 'string' ? w : (w?.word || '')).filter(Boolean);
          }
        } catch {
          importedWords = text.split('\n');
        }
      } else {
        importedWords = text.split('\n');
      }

      let addedCount = 0;
      for (const raw of importedWords) {
        const norm = normalize(raw.trim());
        if (norm && !this.customWordsList.includes(norm)) {
          this.customWordsList.push(norm);
          spellChecker.addWord(norm);
          this.sessionIgnoredWords.add(norm);
          this.markWordAsCorrectInSession(norm);
          addedCount++;
        }
      }

      if (addedCount > 0) {
        localStorage.setItem('claws_custom_words', JSON.stringify(this.customWordsList));
        this.updateCustomWordsCountBadge();
        this.renderCustomWordsList();
        this.showToast(`Successfully imported ${addedCount} new custom words`);
      } else {
        this.showToast('No new words found in file');
      }
    } catch (e) {
      console.error('Failed to import custom words file:', e);
      this.showToast('Failed to import file');
    }
  }

  private ignoreWordForSession(word: string): void {
    this.sessionIgnoredWords.add(word);
    this.markWordAsCorrectInSession(word);
    this.hidePopover();
    this.showToast(`Ignored "${word}" for this session`);
  }

  private markWordAsCorrectInSession(word: string): void {
    let typoCount = 0;
    this.currentTokens.forEach(t => {
      if (t.token === word) {
        t.spellResult = {
          word: t.token,
          normalized_word: t.token,
          is_correct: true,
          suggestions: []
        };
      } else if (t.spellResult && !t.spellResult.is_correct) {
        typoCount++;
      }
    });

    if (this.toggleSpellcheck?.checked && typoCount > 0) {
      this.spellStats.style.display = 'inline-block';
      this.spellStats.textContent = `${typoCount} typo${typoCount > 1 ? 's' : ''}`;
    } else {
      this.spellStats.style.display = 'none';
    }

    this.renderOutput();
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
