@echo off
chcp 65001 > nul

echo ========================================
echo CAI DAT THU VIEN
echo ========================================
python -m pip install -r requirements.txt

echo.
echo ========================================
echo TAI MO HINH
echo ========================================
python download_models.py

echo.
echo Hoan tat buoc cai dat.
echo Tiep theo hay chay tung lenh trong README.md
echo ========================================
pause
