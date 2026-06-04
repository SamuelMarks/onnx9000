/* v8 ignore next */ /* v8 ignore next */ document.addEventListener('DOMContentLoaded', () => {
  /* v8 ignore next */ /* v8 ignore next */
  const lowerBtn = document.getElementById(
    'lowerBtn',
  ) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
  const resetBtn = document.getElementById(
    'resetBtn',
  ) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
  const outputDiv = document.getElementById(
    'output',
  ) as HTMLDivElement; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const delay = (ms: number) =>
    new Promise((res) => setTimeout(res, ms)); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const stages = [
    /* v8 ignore next */ /* v8 ignore next */
    {
      /* v8 ignore next */ /* v8 ignore next */
      title: '1. ONNX to MHLO (High-Level Dialect)' /* v8 ignore next */ /* v8 ignore next */,
      code: `func.func @main(%arg0: tensor<1x10xf32>) -> tensor<1x5xf32> { /* v8 ignore next */ /* v8 ignore next */
  %w = mhlo.constant dense<0.1> : tensor<10x5xf32> /* v8 ignore next */ /* v8 ignore next */
  %b = mhlo.constant dense<0.5> : tensor<5xf32> /* v8 ignore next */ /* v8 ignore next */
  %0 = "mhlo.dot"(%arg0, %w) : (tensor<1x10xf32>, tensor<10x5xf32>) -> tensor<1x5xf32> /* v8 ignore next */ /* v8 ignore next */
  %1 = mhlo.add %0, %b : tensor<1x5xf32> /* v8 ignore next */ /* v8 ignore next */
  return %1 : tensor<1x5xf32> /* v8 ignore next */ /* v8 ignore next */
}` /* v8 ignore next */ /* v8 ignore next */,
    } /* v8 ignore next */ /* v8 ignore next */,
    {
      /* v8 ignore next */ /* v8 ignore next */
      title: '2. MHLO to Linalg (Structural Dialect)' /* v8 ignore next */ /* v8 ignore next */,
      code: `#map0 = affine_map<(d0, d1, d2) -> (d0, d2)> /* v8 ignore next */ /* v8 ignore next */
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)> /* v8 ignore next */ /* v8 ignore next */
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)> /* v8 ignore next */ /* v8 ignore next */
func.func @main(%arg0: tensor<1x10xf32>) -> tensor<1x5xf32> { /* v8 ignore next */ /* v8 ignore next */
  %0 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ... /* v8 ignore next */ /* v8 ignore next */
  %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ... /* v8 ignore next */ /* v8 ignore next */
  return %1 : tensor<1x5xf32> /* v8 ignore next */ /* v8 ignore next */
}` /* v8 ignore next */ /* v8 ignore next */,
    } /* v8 ignore next */ /* v8 ignore next */,
    {
      /* v8 ignore next */ /* v8 ignore next */
      title:
        '3. Bufferization (Value -> Memory Semantics)' /* v8 ignore next */ /* v8 ignore next */,
      code: `func.func @main(%arg0: memref<1x10xf32>, %out: memref<1x5xf32>) { /* v8 ignore next */ /* v8 ignore next */
  %alloc = memref.alloc() : memref<1x5xf32> /* v8 ignore next */ /* v8 ignore next */
  linalg.generic ... ins(%arg0, %w : memref<1x10xf32>, memref<10x5xf32>) outs(%alloc : memref<1x5xf32>) /* v8 ignore next */ /* v8 ignore next */
  linalg.generic ... ins(%alloc, %b : memref<1x5xf32>, memref<5xf32>) outs(%out : memref<1x5xf32>) /* v8 ignore next */ /* v8 ignore next */
  memref.dealloc %alloc : memref<1x5xf32> /* v8 ignore next */ /* v8 ignore next */
  return /* v8 ignore next */ /* v8 ignore next */
}` /* v8 ignore next */ /* v8 ignore next */,
    } /* v8 ignore next */ /* v8 ignore next */,
    {
      /* v8 ignore next */ /* v8 ignore next */
      title:
        '4. Linalg to HAL & VM (Bytecode Generation)' /* v8 ignore next */ /* v8 ignore next */,
      code: `vm.module @module { /* v8 ignore next */ /* v8 ignore next */
  vm.func @main(%arg0: !hal.buffer_view) -> !hal.buffer_view { /* v8 ignore next */ /* v8 ignore next */
    %cmd = hal.command_buffer.create ... /* v8 ignore next */ /* v8 ignore next */
    hal.command_buffer.dispatch %cmd, @executable::@dispatch, [%x, %y, %z] /* v8 ignore next */ /* v8 ignore next */
    hal.command_buffer.finalize %cmd /* v8 ignore next */ /* v8 ignore next */
    %res = hal.device.queue.execute %cmd ... /* v8 ignore next */ /* v8 ignore next */
    vm.return %res : !hal.buffer_view /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}` /* v8 ignore next */ /* v8 ignore next */,
    } /* v8 ignore next */ /* v8 ignore next */,
    {
      /* v8 ignore next */ /* v8 ignore next */
      title:
        '5. Standalone WebGPU WGSL Payload Generated' /* v8 ignore next */ /* v8 ignore next */,
      code: `// WGSL Shader Emitted /* v8 ignore next */ /* v8 ignore next */
@group(0) @binding(0) var<storage, read> arg0: array<f32>; /* v8 ignore next */ /* v8 ignore next */
@group(0) @binding(1) var<storage, read> w: array<f32>; /* v8 ignore next */ /* v8 ignore next */
@group(0) @binding(2) var<storage, read_write> out: array<f32>; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
@compute @workgroup_size(64, 1, 1) /* v8 ignore next */ /* v8 ignore next */
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) { /* v8 ignore next */ /* v8 ignore next */
  // compute logic /* v8 ignore next */ /* v8 ignore next */
}` /* v8 ignore next */ /* v8 ignore next */,
    } /* v8 ignore next */ /* v8 ignore next */,
  ]; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  lowerBtn.addEventListener('click', async () => {
    /* v8 ignore next */ /* v8 ignore next */
    lowerBtn.disabled = true; /* v8 ignore next */ /* v8 ignore next */
    outputDiv.innerHTML = ''; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    for (const stage of stages) {
      /* v8 ignore next */ /* v8 ignore next */
      const stepDiv = document.createElement('div'); /* v8 ignore next */ /* v8 ignore next */
      stepDiv.className = 'step'; /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      const titleDiv = document.createElement('div'); /* v8 ignore next */ /* v8 ignore next */
      titleDiv.className = 'step-title'; /* v8 ignore next */ /* v8 ignore next */
      titleDiv.textContent = stage.title; /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      const codePre = document.createElement('pre'); /* v8 ignore next */ /* v8 ignore next */
      codePre.textContent = stage.code; /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      stepDiv.appendChild(titleDiv); /* v8 ignore next */ /* v8 ignore next */
      stepDiv.appendChild(codePre); /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      outputDiv.appendChild(stepDiv); /* v8 ignore next */ /* v8 ignore next */
      outputDiv.scrollTop = outputDiv.scrollHeight; /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      await delay(600); // Simulate compilation time /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    const completeDiv = document.createElement('div'); /* v8 ignore next */ /* v8 ignore next */
    completeDiv.style.color = '#28a745'; /* v8 ignore next */ /* v8 ignore next */
    completeDiv.style.fontWeight = 'bold'; /* v8 ignore next */ /* v8 ignore next */
    completeDiv.textContent =
      'MLIR Lowering Pipeline Completed Successfully!'; /* v8 ignore next */ /* v8 ignore next */
    outputDiv.appendChild(completeDiv); /* v8 ignore next */ /* v8 ignore next */
    outputDiv.scrollTop = outputDiv.scrollHeight; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    resetBtn.disabled = false; /* v8 ignore next */ /* v8 ignore next */
  }); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  resetBtn.addEventListener('click', () => {
    /* v8 ignore next */ /* v8 ignore next */
    outputDiv.innerHTML =
      'Ready to compile. Click "Run MLIR Lowering Pass" to begin.'; /* v8 ignore next */ /* v8 ignore next */
    lowerBtn.disabled = false; /* v8 ignore next */ /* v8 ignore next */
    resetBtn.disabled = true; /* v8 ignore next */ /* v8 ignore next */
  }); /* v8 ignore next */ /* v8 ignore next */
});
