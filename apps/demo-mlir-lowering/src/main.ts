document.addEventListener('DOMContentLoaded', () => {
  const lowerBtn = document.getElementById('lowerBtn') as HTMLButtonElement;
  const resetBtn = document.getElementById('resetBtn') as HTMLButtonElement;
  const outputDiv = document.getElementById('output') as HTMLDivElement;

  const delay = (ms: number) => new Promise((res) => setTimeout(res, ms));

  const stages = [
    {
      title: '1. ONNX to MHLO (High-Level Dialect)',
      code: `func.func @main(%arg0: tensor<1x10xf32>) -> tensor<1x5xf32> {
  %w = mhlo.constant dense<0.1> : tensor<10x5xf32>
  %b = mhlo.constant dense<0.5> : tensor<5xf32>
  %0 = "mhlo.dot"(%arg0, %w) : (tensor<1x10xf32>, tensor<10x5xf32>) -> tensor<1x5xf32>
  %1 = mhlo.add %0, %b : tensor<1x5xf32>
  return %1 : tensor<1x5xf32>
}`,
    },
    {
      title: '2. MHLO to Linalg (Structural Dialect)',
      code: `#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @main(%arg0: tensor<1x10xf32>) -> tensor<1x5xf32> {
  %0 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ...
  %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ...
  return %1 : tensor<1x5xf32>
}`,
    },
    {
      title: '3. Bufferization (Value -> Memory Semantics)',
      code: `func.func @main(%arg0: memref<1x10xf32>, %out: memref<1x5xf32>) {
  %alloc = memref.alloc() : memref<1x5xf32>
  linalg.generic ... ins(%arg0, %w : memref<1x10xf32>, memref<10x5xf32>) outs(%alloc : memref<1x5xf32>)
  linalg.generic ... ins(%alloc, %b : memref<1x5xf32>, memref<5xf32>) outs(%out : memref<1x5xf32>)
  memref.dealloc %alloc : memref<1x5xf32>
  return
}`,
    },
    {
      title: '4. Linalg to HAL & VM (Bytecode Generation)',
      code: `vm.module @module {
  vm.func @main(%arg0: !hal.buffer_view) -> !hal.buffer_view {
    %cmd = hal.command_buffer.create ...
    hal.command_buffer.dispatch %cmd, @executable::@dispatch, [%x, %y, %z]
    hal.command_buffer.finalize %cmd
    %res = hal.device.queue.execute %cmd ...
    vm.return %res : !hal.buffer_view
  }
}`,
    },
    {
      title: '5. Standalone WebGPU WGSL Payload Generated',
      code: `// WGSL Shader Emitted
@group(0) @binding(0) var<storage, read> arg0: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // compute logic
}`,
    },
  ];

  lowerBtn.addEventListener('click', async () => {
    lowerBtn.disabled = true;
    outputDiv.innerHTML = '';

    for (const stage of stages) {
      const stepDiv = document.createElement('div');
      stepDiv.className = 'step';

      const titleDiv = document.createElement('div');
      titleDiv.className = 'step-title';
      titleDiv.textContent = stage.title;

      const codePre = document.createElement('pre');
      codePre.textContent = stage.code;

      stepDiv.appendChild(titleDiv);
      stepDiv.appendChild(codePre);

      outputDiv.appendChild(stepDiv);
      outputDiv.scrollTop = outputDiv.scrollHeight;

      await delay(600); // Simulate compilation time
    }

    const completeDiv = document.createElement('div');
    completeDiv.style.color = '#28a745';
    completeDiv.style.fontWeight = 'bold';
    completeDiv.textContent = 'MLIR Lowering Pipeline Completed Successfully!';
    outputDiv.appendChild(completeDiv);
    outputDiv.scrollTop = outputDiv.scrollHeight;

    resetBtn.disabled = false;
  });

  resetBtn.addEventListener('click', () => {
    outputDiv.innerHTML = 'Ready to compile. Click "Run MLIR Lowering Pass" to begin.';
    lowerBtn.disabled = false;
    resetBtn.disabled = true;
  });
});
