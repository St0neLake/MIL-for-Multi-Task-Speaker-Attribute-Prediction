@echo off
REM Enable delayed expansion so we can do string manipulations in a FOR loop
setlocal enabledelayedexpansion

REM Get the directory of this script (with trailing backslash).
REM ~dp0 includes the full drive letter and path of this script’s location.
set "SCRIPT_DIR=%~dp0"

REM Build paths relative to the script directory
set "hein_daily_raw_data_dir=%SCRIPT_DIR%hein-daily"
echo hein_daily_raw_data_dir: %hein_daily_raw_data_dir%

set "congress_terms_raw_data_address=%SCRIPT_DIR%congress-terms.csv"
echo congress_terms_raw_data_address: %congress_terms_raw_data_address%

set "preprocessed_data_dir=%SCRIPT_DIR%preprocessed_data"
echo preprocessed_data_dir: %preprocessed_data_dir%

REM Get the grandparent directory of the script, then append data\political_data.csv
set "GRANDPARENT_DIR=%SCRIPT_DIR%..\.."
for %%I in ("%GRANDPARENT_DIR%") do set "GRANDPARENT_DIR=%%~fI"

set "final_data_address=%GRANDPARENT_DIR%\data\political_data.csv"
echo final_data_address: %final_data_address%

REM Collect the numeric part of each "speeches_XXXX.txt" filename under hein_daily_raw_data_dir
set "years="
for %%f in ("%hein_daily_raw_data_dir%\speeches_*.txt") do (
    REM Display each matching file name for debugging
    echo Found: %%~nxf

    REM %%~nf is the filename without path and extension, e.g. "speeches_114"
    set "filename=%%~nf"
    REM Substring from index 9 onward (because "speeches_" is 9 characters)
    set "year=!filename:~9!"

    if defined years (
        REM Append to existing list using delayed expansion
        set "years=!years! !year!"
    ) else (
        REM First value
        set "years=!year!"
    )
)

REM Show final aggregated years just to confirm
echo Final years list: %years%

REM Run the Python script with the same arguments as the original Bash script
python preprocess.py ^
  --hein_daily_raw_data_dir "%hein_daily_raw_data_dir%" ^
  --congress_terms_raw_data_address "%congress_terms_raw_data_address%" ^
  --hein_daily_preprocessed_data_dir "%preprocessed_data_dir%" ^
  --final_data_address "%final_data_address%" ^
  --years %years%

endlocal
