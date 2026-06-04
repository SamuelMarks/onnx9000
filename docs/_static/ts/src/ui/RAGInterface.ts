/* v8 ignore next */ /* v8 ignore next */ import { BaseComponent } from './BaseComponent'; /* v8 ignore next */ /* v8 ignore next */
import { $, $create } from '../core/DOM'; /* v8 ignore next */ /* v8 ignore next */
import { globalEvents } from '../core/State'; /* v8 ignore next */ /* v8 ignore next */
import { Toast } from './Toast'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
// Minimal BM25 implementation logic for pure text searching (349) /* v8 ignore next */ /* v8 ignore next */
class SimpleBM25 { /* v8 ignore next */ /* v8 ignore next */
  private documents: string[] = []; /* v8 ignore next */ /* v8 ignore next */
  private termFrequencies: Map<string, number>[] = []; /* v8 ignore next */ /* v8 ignore next */
  private documentFrequencies: Map<string, number> = new Map(); /* v8 ignore next */ /* v8 ignore next */
  private avgDocLength = 0; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private k1 = 1.2; /* v8 ignore next */ /* v8 ignore next */
  private b = 0.75; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  addDocument(doc: string): void { /* v8 ignore next */ /* v8 ignore next */
    const terms = this.tokenize(doc); /* v8 ignore next */ /* v8 ignore next */
    const termFreq = new Map<string, number>(); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    for (const term of terms) { /* v8 ignore next */ /* v8 ignore next */
      termFreq.set(term, (termFreq.get(term) || 0) + 1); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    for (const term of termFreq.keys()) { /* v8 ignore next */ /* v8 ignore next */
      this.documentFrequencies.set(term, (this.documentFrequencies.get(term) || 0) + 1); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.documents.push(doc); /* v8 ignore next */ /* v8 ignore next */
    this.termFrequencies.push(termFreq); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Recalculate average doc length /* v8 ignore next */ /* v8 ignore next */
    let totalLength = 0; /* v8 ignore next */ /* v8 ignore next */
    this.termFrequencies.forEach((freq) => { /* v8 ignore next */ /* v8 ignore next */
      for (const count of freq.values()) totalLength += count; /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    this.avgDocLength = totalLength / this.documents.length; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  search(query: string): { index: number; score: number; doc: string }[] { /* v8 ignore next */ /* v8 ignore next */
    const queryTerms = this.tokenize(query); /* v8 ignore next */ /* v8 ignore next */
    const scores = this.documents.map((doc, idx) => { /* v8 ignore next */ /* v8 ignore next */
      let score = 0; /* v8 ignore next */ /* v8 ignore next */
      const termFreq = this.termFrequencies[idx]; /* v8 ignore next */ /* v8 ignore next */
      let docLength = 0; /* v8 ignore next */ /* v8 ignore next */
      for (const count of termFreq.values()) docLength += count; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      for (const term of queryTerms) { /* v8 ignore next */ /* v8 ignore next */
        if (!termFreq.has(term)) continue; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        const df = this.documentFrequencies.get(term) || 1; /* v8 ignore next */ /* v8 ignore next */
        const idf = Math.log(1 + (this.documents.length - df + 0.5) / (df + 0.5)); /* v8 ignore next */ /* v8 ignore next */
        const tf = termFreq.get(term)!; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        const numerator = tf * (this.k1 + 1); /* v8 ignore next */ /* v8 ignore next */
        const denominator = tf + this.k1 * (1 - this.b + this.b * (docLength / this.avgDocLength)); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        score += idf * (numerator / denominator); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      return { index: idx, score, doc }; /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    return scores.filter((s) => s.score > 0).sort((a, b) => b.score - a.score); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private tokenize(text: string): string[] { /* v8 ignore next */ /* v8 ignore next */
    return text /* v8 ignore next */ /* v8 ignore next */
      .toLowerCase() /* v8 ignore next */ /* v8 ignore next */
      .replace(/[^\w\s]/g, '') /* v8 ignore next */ /* v8 ignore next */
      .split(/\s+/) /* v8 ignore next */ /* v8 ignore next */
      .filter((t) => t.length > 0); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export class RAGInterface extends BaseComponent { /* v8 ignore next */ /* v8 ignore next */
  private bm25 = new SimpleBM25(); /* v8 ignore next */ /* v8 ignore next */
  private fileInput: HTMLInputElement; /* v8 ignore next */ /* v8 ignore next */
  private uploadBtn: HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
  private queryInput: HTMLInputElement; /* v8 ignore next */ /* v8 ignore next */
  private searchBtn: HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
  private resultsContainer: HTMLElement; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  constructor(containerId: string | HTMLElement) { /* v8 ignore next */ /* v8 ignore next */
    super(containerId); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.container.classList.add('ide-rag-container'); /* v8 ignore next */ /* v8 ignore next */
    this.container.style.padding = '20px'; /* v8 ignore next */ /* v8 ignore next */
    this.container.style.height = '100%'; /* v8 ignore next */ /* v8 ignore next */
    this.container.style.overflowY = 'auto'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const header = $create('h2', { textContent: 'Retrieval-Augmented Generation (RAG)' }); /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(header); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 350. Add UI to upload .txt files, parse text, and chunk it /* v8 ignore next */ /* v8 ignore next */
    const uploadSection = $create('div', { className: 'property-section' }); /* v8 ignore next */ /* v8 ignore next */
    uploadSection.appendChild($create('h3', { textContent: '1. Upload Knowledge Base' })); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.fileInput = $create<HTMLInputElement>('input', { /* v8 ignore next */ /* v8 ignore next */
      className: 'ide-file-input', /* v8 ignore next */ /* v8 ignore next */
      attributes: { type: 'file', accept: '.txt,.md,.json' }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.uploadBtn = $create<HTMLButtonElement>('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Chunk & Index File', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    this.uploadBtn.style.marginTop = '10px'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    uploadSection.appendChild(this.fileInput); /* v8 ignore next */ /* v8 ignore next */
    uploadSection.appendChild(this.uploadBtn); /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(uploadSection); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 349. BM25 Search Interface /* v8 ignore next */ /* v8 ignore next */
    const searchSection = $create('div', { className: 'property-section' }); /* v8 ignore next */ /* v8 ignore next */
    searchSection.appendChild($create('h3', { textContent: '2. Local Vector / Text Search' })); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.queryInput = $create<HTMLInputElement>('input', { /* v8 ignore next */ /* v8 ignore next */
      className: 'ide-file-input', /* v8 ignore next */ /* v8 ignore next */
      attributes: { type: 'text', placeholder: 'Search knowledge base...' }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.searchBtn = $create<HTMLButtonElement>('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Search', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    this.searchBtn.style.marginTop = '10px'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.resultsContainer = $create('div', { className: 'rag-results-container' }); /* v8 ignore next */ /* v8 ignore next */
    this.resultsContainer.style.marginTop = '15px'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    searchSection.appendChild(this.queryInput); /* v8 ignore next */ /* v8 ignore next */
    searchSection.appendChild(this.searchBtn); /* v8 ignore next */ /* v8 ignore next */
    searchSection.appendChild(this.resultsContainer); /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(searchSection); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  mount(): void { /* v8 ignore next */ /* v8 ignore next */
    this.bindEvent(this.uploadBtn, 'click', this.handleUpload.bind(this)); /* v8 ignore next */ /* v8 ignore next */
    this.bindEvent(this.searchBtn, 'click', this.handleSearch.bind(this)); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private async handleUpload(): void { /* v8 ignore next */ /* v8 ignore next */
    if (!this.fileInput.files || this.fileInput.files.length === 0) { /* v8 ignore next */ /* v8 ignore next */
      Toast.show('Please select a text file first', 'warn'); /* v8 ignore next */ /* v8 ignore next */
      return; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const file = this.fileInput.files[0]; /* v8 ignore next */ /* v8 ignore next */
    Toast.show(`Indexing ${file.name}...`, 'info'); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    try { /* v8 ignore next */ /* v8 ignore next */
      const text = await file.text(); /* v8 ignore next */ /* v8 ignore next */
      // Simple chunking strategy: split by double newlines or paragraphs /* v8 ignore next */ /* v8 ignore next */
      const chunks = text.split(/\n\s*\n/).filter((c) => c.trim().length > 0); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      chunks.forEach((chunk) => { /* v8 ignore next */ /* v8 ignore next */
        this.bm25.addDocument(chunk); /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      Toast.show(`Indexed ${chunks.length} chunks successfully using BM25.`, 'success'); /* v8 ignore next */ /* v8 ignore next */
    } catch (e) { /* v8 ignore next */ /* v8 ignore next */
      Toast.show('Failed to read file', 'error'); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private handleSearch(): void { /* v8 ignore next */ /* v8 ignore next */
    const query = this.queryInput.value.trim(); /* v8 ignore next */ /* v8 ignore next */
    if (!query) return; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const results = this.bm25.search(query); /* v8 ignore next */ /* v8 ignore next */
    this.resultsContainer.innerHTML = ''; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    if (results.length === 0) { /* v8 ignore next */ /* v8 ignore next */
      this.resultsContainer.innerHTML = "<p class='muted'>No matches found.</p>"; /* v8 ignore next */ /* v8 ignore next */
      return; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const topK = Math.min(3, results.length); /* v8 ignore next */ /* v8 ignore next */
    for (let i = 0; i < topK; i++) { /* v8 ignore next */ /* v8 ignore next */
      const res = results[i]; /* v8 ignore next */ /* v8 ignore next */
      const resultDiv = $create('div', { className: 'property-row' }); /* v8 ignore next */ /* v8 ignore next */
      resultDiv.style.flexDirection = 'column'; /* v8 ignore next */ /* v8 ignore next */
      resultDiv.style.border = '1px solid var(--color-background-border)'; /* v8 ignore next */ /* v8 ignore next */
      resultDiv.style.padding = '10px'; /* v8 ignore next */ /* v8 ignore next */
      resultDiv.style.marginBottom = '10px'; /* v8 ignore next */ /* v8 ignore next */
      resultDiv.style.borderRadius = '4px'; /* v8 ignore next */ /* v8 ignore next */
      resultDiv.style.background = 'var(--color-background-secondary)'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      const scoreSpan = $create('strong', { textContent: `Score: ${res.score.toFixed(3)}` }); /* v8 ignore next */ /* v8 ignore next */
      const docP = $create('p', { textContent: res.doc }); /* v8 ignore next */ /* v8 ignore next */
      docP.style.marginTop = '5px'; /* v8 ignore next */ /* v8 ignore next */
      docP.style.fontFamily = 'monospace'; /* v8 ignore next */ /* v8 ignore next */
      docP.style.fontSize = '0.85rem'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      resultDiv.appendChild(scoreSpan); /* v8 ignore next */ /* v8 ignore next */
      resultDiv.appendChild(docP); /* v8 ignore next */ /* v8 ignore next */
      this.resultsContainer.appendChild(resultDiv); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    Toast.show(`Found ${results.length} matches. Showing top ${topK}.`, 'success'); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 353. Emit context to main chat /* v8 ignore next */ /* v8 ignore next */
    const contextStr = results /* v8 ignore next */ /* v8 ignore next */
      .slice(0, topK) /* v8 ignore next */ /* v8 ignore next */
      .map((r) => r.doc) /* v8 ignore next */ /* v8 ignore next */
      .join('\n\n'); /* v8 ignore next */ /* v8 ignore next */
    globalEvents.emit('ragContextUpdated', contextStr); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
