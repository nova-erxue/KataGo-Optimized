@echo off
setlocal DisableDelayedExpansion
set "PATH=%~dp0lib;%PATH%"
if defined CUDA_PATH_V13_2 set "PATH=%CUDA_PATH_V13_2%\bin;%PATH%"
if defined KATAGO_LIBRARY_PATH set "PATH=%KATAGO_LIBRARY_PATH%;%PATH%"
"%~dp0katago.exe" %*
exit /b %errorlevel%
