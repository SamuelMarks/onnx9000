.PHONY: docs clean build_wasm

build_wasm:
	@echo "Building ONNX9000 WebAssembly engine..."
	pnpm exec asc packages/js/core/src/wasm/engine.ts -O -o docs/html_extra/onnx9000.wasm

docs: build_wasm
	@echo "Generating Typedoc Markdown for JS API..."
	export npm_config_yes=true && npx typedoc --options typedoc.json
	@echo "Building Sphinx documentation..."
	uv run sphinx-build -M html docs docs/_build
	@echo "Documentation built in docs/_build/html"

clean:
	rm -rf docs/_build
	rm -rf docs/js-api
