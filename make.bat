@ECHO OFF

IF "%1" == "build_wasm" (
    ECHO Building ONNX9000 WebAssembly engine...
    CALL pnpm exec asc packages\js\core\src\wasm\engine.ts -O -o docs\html_extra\onnx9000.wasm
    GOTO END
)

IF "%1" == "docs" (
    ECHO Building ONNX9000 WebAssembly engine...
    CALL pnpm exec asc packages\js\core\src\wasm\engine.ts -O -o docs\html_extra\onnx9000.wasm
    ECHO Generating Typedoc Markdown for JS API...
    SET npm_config_yes=true
    CALL npx typedoc --options typedoc.json
    ECHO Building Sphinx documentation...
    CALL uv run sphinx-build -M html docs docs\_build
    ECHO Documentation built in docs\_build\html
    GOTO END
)

IF "%1" == "clean" (
    ECHO Cleaning docs build directory...
    RMDIR /S /Q docs\_build
    RMDIR /S /Q docs\js-api
    GOTO END
)

ECHO Please specify a target, e.g., "make docs", "make build_wasm" or "make clean"

:END
