@echo off

REM Percorsi
set SERVER_PATH=%cd%\Test
set PLAYER_PATH=%SERVER_PATH%\..\LoDaLe_tablut

REM Apri una finestra per il server
echo Starting the server...
start "Server" cmd /k "cd /d %SERVER_PATH% && java -jar Server.jar"

REM Attendi che il server si inizializzi
timeout /t 1 >nul

REM Apri una finestra per il client WHITE
echo Starting the WHITE client...
start "Client WHITE" cmd /k "cd /d %PLAYER_PATH% && python PLAYER.py WHITE 60 127.0.0.1 search flag_2"

REM Apri una finestra per il client BLACK
echo Starting the BLACK client...
start "Client BLACK" cmd /k "cd /d %PLAYER_PATH% && python PLAYER.py BLACK 60 127.0.0.1 search grey"

echo All processes started!
