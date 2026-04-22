@echo off
setlocal

REM Usage:
REM   combile.bat            -> compiles paper.tex
REM   combile.bat myfile.tex -> compiles myfile.tex

set "TEXFILE=%~1"
if "%TEXFILE%"=="" set "TEXFILE=paper.tex"

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

if not exist "%TEXFILE%" (
    if exist "%SCRIPT_DIR%\%TEXFILE%" (
        set "TEXFILE=%SCRIPT_DIR%\%TEXFILE%"
    )
)

if not exist "%TEXFILE%" (
    echo [ERROR] File not found: %TEXFILE%
    exit /b 1
)

for %%I in ("%TEXFILE%") do (
    set "TEXFILE=%%~fI"
    set "TEXDIR=%%~dpI"
    set "TEXNAME=%%~nxI"
    set "TEXROOT=%%~nI"
)

pushd "%TEXDIR%" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Cannot access directory: %TEXDIR%
    exit /b 1
)

echo [INFO] Compiling %TEXNAME%

where latexmk >nul 2>nul
if not errorlevel 1 (
    echo [INFO] Using latexmk...
    latexmk -xelatex -interaction=nonstopmode -synctex=1 "%TEXNAME%"
    if errorlevel 1 (
        echo [ERROR] latexmk failed.
        popd
        exit /b 1
    )
) else (
    where xelatex >nul 2>nul
    if not errorlevel 1 (
        echo [INFO] latexmk not found. Falling back to xelatex ^(3 passes^)...
        xelatex -interaction=nonstopmode "%TEXNAME%"
        if errorlevel 1 (
            popd
            exit /b 1
        )

        xelatex -interaction=nonstopmode "%TEXNAME%"
        if errorlevel 1 (
            popd
            exit /b 1
        )

        xelatex -interaction=nonstopmode "%TEXNAME%"
        if errorlevel 1 (
            popd
            exit /b 1
        )
    ) else (
        echo [INFO] xelatex not found. Falling back to pdflatex ^(3 passes^)...
        pdflatex -interaction=nonstopmode "%TEXNAME%"
        if errorlevel 1 (
            popd
            exit /b 1
        )

        pdflatex -interaction=nonstopmode "%TEXNAME%"
        if errorlevel 1 (
            popd
            exit /b 1
        )

        pdflatex -interaction=nonstopmode "%TEXNAME%"
        if errorlevel 1 (
            popd
            exit /b 1
        )
    )
)

echo [SUCCESS] PDF build completed.
if exist "%TEXROOT%.pdf" start "" "%TEXROOT%.pdf" >nul 2>nul
popd
endlocal
exit /b 0
