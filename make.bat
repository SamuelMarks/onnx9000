@ECHO OFF

IF "%1" == "docs" (
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

ECHO Please specify a target, e.g., "make docs" or "make clean"

:END
