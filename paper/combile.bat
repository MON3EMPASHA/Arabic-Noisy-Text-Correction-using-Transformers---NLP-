@echo off
setlocal

REM Usage:
REM   combile.bat            -> compiles paper.tex
REM   combile.bat myfile.tex -> compiles myfile.tex

set "TEXFILE=%~1"
if "%TEXFILE%"=="" set "TEXFILE=paper.tex"
for %%I in ("%TEXFILE%") do set "TEXROOT=%%~nI"

if not exist "%TEXFILE%" (
    echo [ERROR] File not found: %TEXFILE%
    exit /b 1
)

echo [INFO] Compiling %TEXFILE%

where latexmk >nul 2>nul
if not errorlevel 1 (
    echo [INFO] Using latexmk...
    latexmk -xelatex -interaction=nonstopmode -synctex=1 "%TEXFILE%"
    if errorlevel 1 (
        echo [ERROR] latexmk failed.
        exit /b 1
    )
) else (
    where xelatex >nul 2>nul
    if not errorlevel 1 (
        echo [INFO] latexmk not found. Falling back to xelatex ^(3 passes^)...
        xelatex -interaction=nonstopmode "%TEXFILE%"
        if errorlevel 1 exit /b 1

        xelatex -interaction=nonstopmode "%TEXFILE%"
        if errorlevel 1 exit /b 1

        xelatex -interaction=nonstopmode "%TEXFILE%"
        if errorlevel 1 exit /b 1
    ) else (
        echo [INFO] xelatex not found. Falling back to pdflatex ^(3 passes^)...
        pdflatex -interaction=nonstopmode "%TEXFILE%"
        if errorlevel 1 exit /b 1

        pdflatex -interaction=nonstopmode "%TEXFILE%"
        if errorlevel 1 exit /b 1

        pdflatex -interaction=nonstopmode "%TEXFILE%"
        if errorlevel 1 exit /b 1
    )
)

for %%F in (
    "paper.aux"
    "paper.log"
    "paper.fls"
    "paper.fdb_latexmk"
    "paper.synctex.gz"
    "paper.toc"
    "paper.out"
    "paper.lof"
    "paper.lot"
    "paper.bbl"
    "paper.blg"
    "paper.xdv"
    "paper.fls"
    "paper.log"
) do if exist "%%~fF" del /q "%%~fF"

echo [SUCCESS] PDF build completed.
endlocal
@REM start "" "%TEXROOT%.pdf"
start paper.pdf
exit /b 0
