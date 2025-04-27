@echo off
setlocal

REM The first three arguments are the number of times to run the script, the IP address, and the port
set "N=%~1"
set "IP=%~2"
set "PORT=%~3"

REM Run 'python CSP_float.py' with the IP address and port
start cmd /K "conda activate torch && python CSP_float.py %IP% %PORT%"

timeout /T 7 /NOBREAK

REM Run 'python ES_float.py' N times with the IP address and port
for /L %%i in (1,1,%N%) do (
    start cmd /K "conda activate torch && python ES_float.py %IP% %PORT%"
)

endlocal