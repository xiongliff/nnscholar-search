@echo off
echo Activating virtual environment...
call nnscholar-search-main\venv\Scripts\activate.bat
echo Installing PyTorch...
pip uninstall torch torchvision torchaudio -y
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.36.0
echo PyTorch installation completed.
pause 