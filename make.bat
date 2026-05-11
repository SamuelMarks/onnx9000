@ECHO OFF

IF "%~1" == "" GOTO build_wasm
IF "%~1" == "build_wasm" GOTO build_wasm
IF "%~1" == "docs" GOTO docs
IF "%~1" == "clean" GOTO clean

ECHO Please specify a target, e.g., "make docs", "make build_wasm" or "make clean"
GOTO end

:build_wasm
ECHO Building ONNX9000 WebAssembly engine...
CALL pnpm exec asc packages\js\core\src\wasm\engine.ts -O -o docs\html_extra\onnx9000.wasm
GOTO end

:docs
ECHO Building ONNX9000 WebAssembly engine...
CALL pnpm exec asc packages\js\core\src\wasm\engine.ts -O -o docs\html_extra\onnx9000.wasm
ECHO Generating Typedoc Markdown for JS API...
CALL pnpm exec typedoc --options typedoc.json
ECHO Building Sphinx documentation...
CALL uv run sphinx-build -M html docs docs\_build
ECHO Documentation built in docs\_build\html
GOTO end

:clean
ECHO Cleaning docs build directory...
IF EXIST docs\_build RMDIR /S /Q docs\_build
IF EXIST docs\js-api RMDIR /S /Q docs\js-api
GOTO end

:end
