@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-moe-u3.ps1" %*
exit /b %ERRORLEVEL%
