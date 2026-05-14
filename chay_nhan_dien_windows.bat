@echo off
chcp 65001 > nul
python src/recognize_attendance.py --threshold 0.50 --margin 0.06
pause
