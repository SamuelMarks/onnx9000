import { Router } from './router';

export class PrometheusMetrics {
  private requestTotal: number = 0;
  private activeRequests: number = 0;
  private gpuMemoryBytes: number = 0;
  private cpuMemoryBytes: number = 0;
  private kvCacheSizeBytes: number = 0;

  // Histogram buckets
  private durationSecondsSum: number = 0;
  private queueSecondsSum: number = 0;

  public incrementRequests() {
    this.requestTotal++;
  }

  public setActiveRequests(count: number) {
    this.activeRequests = count;
  }

  public setGpuMemoryBytes(bytes: number) {
    this.gpuMemoryBytes = bytes;
  }

  public setCpuMemoryBytes(bytes: number) {
    this.cpuMemoryBytes = bytes;
  }

  public setKvCacheSizeBytes(bytes: number) {
    this.kvCacheSizeBytes = bytes;
  }

  public recordRequestDuration(seconds: number) {
    this.durationSecondsSum += seconds;
  }

  public recordQueueDuration(seconds: number) {
    this.queueSecondsSum += seconds;
  }

  public generateTextFormat(): string {
    let out = '';

    out += '# HELP onnx9000_inference_request_total Total number of inference requests\n';
    out += '# TYPE onnx9000_inference_request_total counter\n';
    out += `onnx9000_inference_request_total ${this.requestTotal}\n\n`;

    out += '# HELP onnx9000_active_requests Number of currently active inference requests\n';
    out += '# TYPE onnx9000_active_requests gauge\n';
    out += `onnx9000_active_requests ${this.activeRequests}\n\n`;

    out += '# HELP onnx9000_gpu_memory_bytes GPU memory usage in bytes\n';
    out += '# TYPE onnx9000_gpu_memory_bytes gauge\n';
    out += `onnx9000_gpu_memory_bytes ${this.gpuMemoryBytes}\n\n`;

    out += '# HELP onnx9000_cpu_memory_bytes CPU memory usage in bytes\n';
    out += '# TYPE onnx9000_cpu_memory_bytes gauge\n';
    out += `onnx9000_cpu_memory_bytes ${this.cpuMemoryBytes}\n\n`;

    out += '# HELP onnx9000_kv_cache_size_bytes KV Cache size in bytes\n';
    out += '# TYPE onnx9000_kv_cache_size_bytes gauge\n';
    out += `onnx9000_kv_cache_size_bytes ${this.kvCacheSizeBytes}\n\n`;

    out += '# HELP onnx9000_inference_request_duration_seconds Total sum of inference duration\n';
    out += '# TYPE onnx9000_inference_request_duration_seconds histogram\n';
    out += `onnx9000_inference_request_duration_seconds_sum ${this.durationSecondsSum}\n\n`;

    out += '# HELP onnx9000_inference_queue_duration_seconds Total sum of queue duration\n';
    out += '# TYPE onnx9000_inference_queue_duration_seconds histogram\n';
    out += `onnx9000_inference_queue_duration_seconds_sum ${this.queueSecondsSum}\n\n`;

    return out;
  }
}

export const globalMetrics = new PrometheusMetrics();

export function addMetricsRoutes(router: Router) {
  // 131. Expose `/metrics` endpoint natively
  router.get('/metrics', async () => {
    // 132. Implement standard Prometheus text-based metrics format.
    return new Response(globalMetrics.generateTextFormat(), {
      status: 200,
      headers: {
        'Content-Type': 'text/plain; version=0.0.4',
      },
    });
  });
}
