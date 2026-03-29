/* eslint-disable */
// @ts-nocheck
export type Language = 'en' | 'es' | 'fr' | 'de' | 'ja';

const translations: Record<Language, Record<string, string>> = {
  en: {
    'lhs.title': 'LHS Container',
    'lhs.run': 'Run Conversion',
    'lhs.converting': 'Converting...',
    'lhs.sources': 'Sources',
    'lhs.files': 'Files',
    'lhs.pipeline': 'Pipeline',
    'rhs.title': 'RHS Container',
    'rhs.run': 'Run Inference',
    'rhs.running': 'Running...',
    'rhs.targets': 'Targets',
    'rhs.optimization': 'Optimization',
    'bottom.console': 'Console',
    'bottom.visualizer': 'ONNX Visualization',
    'bottom.profiler': 'Execution Profiler',
    'wasm.loading': 'Loading AI Compiler Engine...',
    'wasm.desc':
      'Please wait while we initialize the WebAssembly environment and load necessary resources.',
    'wasm.start': 'Start Demo',
    'wasm.error': 'Initialization Failed',
    'tensor.upload': 'Upload File',
    'tensor.generate': 'Generate Dummy Data',
    'tensor.submit': 'Submit Inputs',
    'tensor.close': 'Close',
    'metrics.title': 'Performance Metrics',
    'toast.success': 'Success',
    'toast.error': 'Error',
    'toast.info': 'Info',
    'olive.quant': 'Quantization',
    'olive.shape': 'Shape Inference',
    'olive.fusion': 'Graph Fusion',
    'olive.quant.none': 'None (FP32)',
    'olive.quant.int8': 'Dynamic INT8',
    'promote.button': 'Promote to Source',
    'promote.tooltip':
      'Use this pipeline step output as the new source file for further conversions.',
    'console.clear': 'Clear Console'
  },
  es: {
    'lhs.title': 'Contenedor Izquierdo',
    'lhs.run': 'Ejecutar Conversión',
    'lhs.converting': 'Convirtiendo...',
    'lhs.sources': 'Fuentes',
    'lhs.files': 'Archivos',
    'lhs.pipeline': 'Canalización',
    'rhs.title': 'Contenedor Derecho',
    'rhs.run': 'Ejecutar Inferencia',
    'rhs.running': 'Ejecutando...',
    'rhs.targets': 'Destinos',
    'rhs.optimization': 'Optimización',
    'bottom.console': 'Consola',
    'bottom.visualizer': 'Visualizador',
    'bottom.profiler': 'Perfilador',
    'wasm.loading': 'Cargando Motor de Compilación de IA...',
    'wasm.desc':
      'Por favor espera mientras inicializamos el entorno WebAssembly y cargamos los recursos necesarios.',
    'wasm.start': 'Iniciar Demo',
    'wasm.error': 'Falló la Inicialización',
    'tensor.upload': 'Subir Archivo',
    'tensor.generate': 'Generar Datos Falsos',
    'tensor.submit': 'Enviar Entradas',
    'tensor.close': 'Cerrar',
    'metrics.title': 'Métricas de Rendimiento',
    'toast.success': 'Éxito',
    'toast.error': 'Error',
    'toast.info': 'Información',
    'olive.quant': 'Cuantización',
    'olive.shape': 'Inferencia de Forma',
    'olive.fusion': 'Fusión de Grafos',
    'olive.quant.none': 'Ninguno (FP32)',
    'olive.quant.int8': 'INT8 Dinámico',
    'promote.button': 'Promover a Fuente',
    'promote.tooltip':
      'Usar esta salida del paso de la canalización como el nuevo archivo fuente para conversiones posteriores.',
    'console.clear': 'Limpiar Consola'
  },
  fr: {
    'lhs.title': 'Conteneur Gauche',
    'lhs.run': 'Lancer la Conversion',
    'lhs.converting': 'Conversion en cours...',
    'lhs.sources': 'Sources',
    'lhs.files': 'Fichiers',
    'lhs.pipeline': 'Pipeline',
    'rhs.title': 'Conteneur Droit',
    'rhs.run': "Lancer l'Inférence",
    'rhs.running': 'Exécution...',
    'rhs.targets': 'Cibles',
    'rhs.optimization': 'Optimisation',
    'bottom.console': 'Console',
    'bottom.visualizer': 'Visualiseur',
    'bottom.profiler': 'Profileur',
    'wasm.loading': 'Chargement du Moteur de Compilation IA...',
    'wasm.desc': "Veuillez patienter pendant l'initialisation de l'environnement WebAssembly.",
    'wasm.start': 'Démarrer la Démo',
    'wasm.error': "Échec de l'Initialisation",
    'tensor.upload': 'Téléverser un Fichier',
    'tensor.generate': 'Générer des Données Factices',
    'tensor.submit': 'Soumettre les Entrées',
    'tensor.close': 'Fermer',
    'metrics.title': 'Mesures de Performance',
    'toast.success': 'Succès',
    'toast.error': 'Erreur',
    'toast.info': 'Info',
    'olive.quant': 'Quantification',
    'olive.shape': 'Inférence de Forme',
    'olive.fusion': 'Fusion de Graphes',
    'olive.quant.none': 'Aucune (FP32)',
    'olive.quant.int8': 'INT8 Dynamique',
    'promote.button': 'Promouvoir comme Source',
    'promote.tooltip': 'Utilisez la sortie de cette étape comme nouveau fichier source.',
    'console.clear': 'Effacer la Console'
  },
  de: {
    'lhs.title': 'Linker Container',
    'lhs.run': 'Konvertierung Ausführen',
    'lhs.converting': 'Konvertiere...',
    'lhs.sources': 'Quellen',
    'lhs.files': 'Dateien',
    'lhs.pipeline': 'Pipeline',
    'rhs.title': 'Rechter Container',
    'rhs.run': 'Inferenz Ausführen',
    'rhs.running': 'Ausführen...',
    'rhs.targets': 'Ziele',
    'rhs.optimization': 'Optimierung',
    'bottom.console': 'Konsole',
    'bottom.visualizer': 'Visualisierer',
    'bottom.profiler': 'Profiler',
    'wasm.loading': 'Lade KI-Kompilierungsengine...',
    'wasm.desc': 'Bitte warten Sie, während wir die WebAssembly-Umgebung initialisieren.',
    'wasm.start': 'Demo Starten',
    'wasm.error': 'Initialisierung Fehlgeschlagen',
    'tensor.upload': 'Datei Hochladen',
    'tensor.generate': 'Dummy-Daten Generieren',
    'tensor.submit': 'Eingaben Senden',
    'tensor.close': 'Schließen',
    'metrics.title': 'Leistungsmetriken',
    'toast.success': 'Erfolg',
    'toast.error': 'Fehler',
    'toast.info': 'Info',
    'olive.quant': 'Quantisierung',
    'olive.shape': 'Form-Inferenz',
    'olive.fusion': 'Graphenfusion',
    'olive.quant.none': 'Keine (FP32)',
    'olive.quant.int8': 'Dynamisches INT8',
    'promote.button': 'Zur Quelle Hochstufen',
    'promote.tooltip': 'Verwenden Sie diese Ausgabe als neue Quelldatei.',
    'console.clear': 'Konsole Leeren'
  },
  ja: {
    'lhs.title': '左側コンテナ',
    'lhs.run': '変換を実行',
    'lhs.converting': '変換中...',
    'lhs.sources': 'ソース',
    'lhs.files': 'ファイル',
    'lhs.pipeline': 'パイプライン',
    'rhs.title': '右側コンテナ',
    'rhs.run': '推論を実行',
    'rhs.running': '実行中...',
    'rhs.targets': 'ターゲット',
    'rhs.optimization': '最適化',
    'bottom.console': 'コンソール',
    'bottom.visualizer': 'ビジュアライザ',
    'bottom.profiler': 'プロファイラ',
    'wasm.loading': 'AIコンパイラエンジンを読み込み中...',
    'wasm.desc': 'WebAssembly環境を初期化しています。しばらくお待ちください。',
    'wasm.start': 'デモを開始',
    'wasm.error': '初期化に失敗しました',
    'tensor.upload': 'ファイルをアップロード',
    'tensor.generate': 'ダミーデータを生成',
    'tensor.submit': '入力を送信',
    'tensor.close': '閉じる',
    'metrics.title': 'パフォーマンスメトリクス',
    'toast.success': '成功',
    'toast.error': 'エラー',
    'toast.info': '情報',
    'olive.quant': '量子化',
    'olive.shape': '形状推論',
    'olive.fusion': 'グラフフュージョン',
    'olive.quant.none': 'なし (FP32)',
    'olive.quant.int8': '動的 INT8',
    'promote.button': 'ソースに昇格',
    'promote.tooltip': 'このパイプラインステップの出力を新しいソースファイルとして使用します。',
    'console.clear': 'コンソールをクリア'
  }
};

class I18nManager {
  private static instance: I18nManager;
  private currentLang: Language = 'en';

  private constructor() {
    this.detectLanguage();
  }

  public static getInstance(): I18nManager {
    if (!I18nManager.instance) {
      I18nManager.instance = new I18nManager();
    }
    return I18nManager.instance;
  }

  public setLanguage(lang: Language) {
    if (translations[lang]) {
      this.currentLang = lang;
      localStorage.setItem('onnx9000-demo-lang', lang);
      document.documentElement.lang = lang;
      // Emit event so components can re-render if needed
      import('./EventBus').then(({ globalEventBus }) => {
        globalEventBus.emit('LANGUAGE_CHANGED', lang);
      });
    }
  }

  public getLanguage(): Language {
    return this.currentLang;
  }

  public t(key: string, args?: Record<string, string>): string {
    const langDict = translations[this.currentLang] || translations['en'];
    let text = langDict[key] || translations['en'][key] || key;
    if (args) {
      for (const [k, v] of Object.entries(args)) {
        text = text.replace(new RegExp(`\\{${k}\\}`, 'g'), v);
      }
    }
    return text;
  }

  private detectLanguage() {
    const saved = localStorage.getItem('onnx9000-demo-lang') as Language;
    if (saved && translations[saved]) {
      this.currentLang = saved;
      return;
    }
    const browserLang = navigator.language.split('-')[0] as Language;
    if (translations[browserLang]) {
      this.currentLang = browserLang;
    }
  }
}

export const i18n = I18nManager.getInstance();
export const t = (key: string, args?: Record<string, string>) => i18n.t(key, args);
