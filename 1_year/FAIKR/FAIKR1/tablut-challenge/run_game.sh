#!/bin/bash

SERVER_PATH="$(pwd)/Test"
PLAYER_PATH="$SERVER_PATH/../LoDaLe_tablut"

# Open terminal tab for server
osascript -e "tell application \"Terminal\" to do script \"cd $SERVER_PATH && java -jar Server.jar\""

# Ensure enough time for correct initialization of server 
sleep 1 

# Open terminal tab for client WHITE
osascript -e "tell application \"Terminal\" to do script \"cd $PLAYER_PATH && python3 PLAYER.py WHITE 60 127.0.0.1 search grey\""

# Open terminal tab for client BLACK
osascript -e "tell application \"Terminal\" to do script \"cd $PLAYER_PATH && python3 PLAYER.py BLACK 60 127.0.0.1 search flag-3\""
