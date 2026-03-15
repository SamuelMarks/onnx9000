#include "src/onnx9000/backends/web/wasm_core.cpp"
#include <cassert>
#include <iostream>

using namespace onnx9000;

/// Executes unit tests for the WebAssembly core memory planner and math operations.

int main() {
    WasmMemoryPlanner planner;
    auto offset1 = planner.allocate(10);
    assert(offset1.value() == 0);
    auto offset2 = planner.allocate(16);
    assert(offset2.value() == 16);
    auto offset3 = planner.allocate(20);
    assert(offset3.value() == 32);

    float a[4] = {1.0, 2.0, -1.0, -2.0};
    float b[4] = {2.0, 3.0, 4.0, 5.0};
    float y[4];

    WasmOps::add_f32(a, b, y, 4);
    assert(y[0] == 3.0 && y[1] == 5.0 && y[2] == 3.0 && y[3] == 3.0);

    WasmOps::mul_f32(a, b, y, 4);
    assert(y[0] == 2.0 && y[1] == 6.0 && y[2] == -4.0 && y[3] == -10.0);

    WasmOps::relu_f32(a, y, 4);
    assert(y[0] == 1.0 && y[1] == 2.0 && y[2] == 0.0 && y[3] == 0.0);

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
