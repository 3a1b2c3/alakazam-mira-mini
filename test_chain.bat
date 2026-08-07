@echo off
echo Starting test...
echo Step 1: Call generate_and_eval
call .\generate_and_eval.bat "outputs/wm_ckpt_63000.pth" "63000" "C:\recordings\mira_wds\test\000\dataset_00000\1ea1a4ce-2fdc-44fe-afdd-8019fbacea28_clip00000_c00000.p0.mp4" 24
echo Step 1 returned with errorlevel %ERRORLEVEL%
echo.
echo Step 2: About to call generate_and_eval again
call .\generate_and_eval.bat "outputs/wm_ckpt_63000.pth" "63000" "C:\recordings\mira_wds\test\000\dataset_00000\1ea1a4ce-2fdc-44fe-afdd-8019fbacea28_clip00000_c00001.p0.mp4" 20
echo Step 2 returned with errorlevel %ERRORLEVEL%
echo Done!
