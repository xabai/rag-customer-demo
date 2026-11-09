@echo off
chcp 65001 >nul
cd /d %~dp0
"E:\Anaconda\Anaconda\python.exe" app.py
pause
