/**
 * @fileoverview metrics.ts
 * Provides metrics functionality for the serve package.
 */
import type { Router } from './router';

/** Represents the `PrometheusMetrics` class. */
export class PrometheusMetrics {
  /** Property `requestTotal` of `PrometheusMetrics`. */
  private requestTotal: number = 0;
  /** Property `activeRequests` of `PrometheusMetrics`. */
  private activeRequests: number = 0;
  /** Property `gpuMemoryBytes` of `PrometheusMetrics`. */
  private gpuMemoryBytes: number = 0;
  /** Property `cpuMemoryBytes` of `PrometheusMetrics`. */
  private cpuMemoryBytes: number = 0;
  /** Property `kvCacheSizeBytes` of `PrometheusMetrics`. */
  private kvCacheSizeBytes: number = 0;

  // Histogram buckets
  /** Property `durationSecondsSum` of `PrometheusMetrics`. */
  private durationSecondsSum: number = 0;
  /** Property `queueSecondsSum` of `PrometheusMetrics`. */
  private queueSecondsSum: number = 0;

  /**
   * Executes `incrementRequests` on `PrometheusMetrics`.
   * @returns Result of `incrementRequests`.
   */
  public incrementRequests() {
    this.requestTotal++;
  }

  /**
   * Executes `setActiveRequests` on `PrometheusMetrics`.
   * @param count Parameter `count`.
   * @returns Result of `setActiveRequests`.
   */
  public setActiveRequests(count: number) {
    this.activeRequests = count;
  }

  /**
   * Executes `setGpuMemoryBytes` on `PrometheusMetrics`.
   * @param bytes Parameter `bytes`.
   * @returns Result of `setGpuMemoryBytes`.
   */
  public setGpuMemoryBytes(bytes: number) {
    this.gpuMemoryBytes = bytes;
  }

  /**
   * Executes `setCpuMemoryBytes` on `PrometheusMetrics`.
   * @param bytes Parameter `bytes`.
   * @returns Result of `setCpuMemoryBytes`.
   */
  public setCpuMemoryBytes(bytes: number) {
    this.cpuMemoryBytes = bytes;
  }

  /**
   * Executes `setKvCacheSizeBytes` on `PrometheusMetrics`.
   * @param bytes Parameter `bytes`.
   * @returns Result of `setKvCacheSizeBytes`.
   */
  public setKvCacheSizeBytes(bytes: number) {
    this.kvCacheSizeBytes = bytes;
  }

  /**
   * Executes `recordRequestDuration` on `PrometheusMetrics`.
   * @param seconds Parameter `seconds`.
   * @returns Result of `recordRequestDuration`.
   */
  public recordRequestDuration(seconds: number) {
    this.durationSecondsSum += seconds;
  }

  /**
   * Executes `recordQueueDuration` on `PrometheusMetrics`.
   * @param seconds Parameter `seconds`.
   * @returns Result of `recordQueueDuration`.
   */
  public recordQueueDuration(seconds: number) {
    this.queueSecondsSum += seconds;
  }

  /**
   * Executes `generateTextFormat` on `PrometheusMetrics`.
   * @returns Result of `generateTextFormat`.
   */
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

/**
 * Executes `addMetricsRoutes`.
 * @param router Parameter `router`.
 * @returns Result of `addMetricsRoutes`.
 */
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
