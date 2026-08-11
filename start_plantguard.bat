@echo off
setlocal
title PlantGuard AI Setup and Launcher

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_plantguard.ps1"

if errorlevel 1 (
  echo.
  echo PlantGuard AI could not start. Review the message above.
  pause
)
