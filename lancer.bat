@echo off
chcp 65001 >nul
title Prevision de la consommation electrique
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Installation de l environnement Python ^(une seule fois, quelques minutes^)...
  python -m venv .venv || (echo Python introuvable : installez Python 3.11+ depuis python.org & pause & exit /b 1)
  ".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || (pause & exit /b 1)
)
echo.
echo Etude comparative : 7 modeles, 104 semaines de test (environ 5 minutes)...
pushd src
"..\.venv\Scripts\python.exe" -m prevision.run || (popd & pause & exit /b 1)
popd
echo.
echo Termine. Les graphiques sont dans reports\figures
start "" "reports\figures"
pause
