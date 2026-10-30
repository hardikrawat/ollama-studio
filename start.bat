@echo off
setlocal
cd /d "%~dp0"
set PORT=3300
:check_port
netstat -ano | findstr /r /c:":%PORT% *LISTENING" >nul
if not errorlevel 1 (
    set /a PORT+=1
    goto check_port
)
echo Starting Local Intelligence Workstation on http://localhost:%PORT% ...
start "" http://localhost:%PORT%
python server.py %PORT%
