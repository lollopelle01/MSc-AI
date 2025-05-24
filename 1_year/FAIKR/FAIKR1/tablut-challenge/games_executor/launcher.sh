#!/bin/bash

# Running hyperparameters
NUM_TESTS=1                                     # each run should be deterministic
H_FLAGS=("grey" "flag-1" "flag-2" "flag-3")     # TODO: add more heuristic flags
P_FLAGS=("search" "random")                     # TODO: add more player flags

# Paths
PLAYER_PATH="../LoDaLe_tablut"

# Create and empty dir for the player outputs
OUTPUTS_DIR="./outputs"
if [ -d "$OUTPUTS_DIR" ]; then
    echo "Cleaning outputs directory..."
    rm -R "$OUTPUTS_DIR"
fi
mkdir -p "$OUTPUTS_DIR"

LOG_DIR="./logs"
if [ -d "$LOG_DIR" ]; then
    echo "Cleaning logs directory..."
    rm -R "$LOG_DIR"
fi
mkdir -p "$LOG_DIR"

# Generate a list of unique combinations and calculate the total matches
UNIQUE_COMBINATIONS=()
for ((i = 0; i < ${#P_FLAGS[@]}; i++)); do
    for ((j = 0; j < ${#P_FLAGS[@]}; j++)); do
        player_flag_1="${P_FLAGS[i]}"
        player_flag_2="${P_FLAGS[j]}"

        if [[ "$player_flag_1" == "random" && "$player_flag_2" == "random" ]]; then
            continue
        fi

        # Considera solo i player 'search' per entrambi
        if [[ "$player_flag_1" == "search" && "$player_flag_2" == "search" ]]; then
            for ((k = 0; k < ${#H_FLAGS[@]}; k++)); do
                for ((l = 0; l < ${#H_FLAGS[@]}; l++)); do
                    # Controlla che le euristiche siano diverse
                    if [ "$k" -ne "$l" ]; then
                        heuristic_1="${H_FLAGS[k]}"
                        heuristic_2="${H_FLAGS[l]}"

                        # Random ha l'euristica 'X'
                        if [[ "$player_flag_1" == "random" ]]; then
                            heuristic_1="X"
                        fi
                        if [[ "$player_flag_2" == "random" ]]; then
                            heuristic_2="X"
                        fi

                        COMBINATION_KEY="${player_flag_1}_${heuristic_1}_vs_${player_flag_2}_${heuristic_2}"
                        if [[ ! " ${UNIQUE_COMBINATIONS[@]} " =~ " ${COMBINATION_KEY} " ]]; then
                            UNIQUE_COMBINATIONS+=("$COMBINATION_KEY")
                        fi
                    fi
                done
            done
        else
            # Caso in cui uno dei player è random, genera combinazioni di euristiche diverse
            for ((k = 0; k < ${#H_FLAGS[@]}; k++)); do
                heuristic_1="${H_FLAGS[k]}"
                if [[ "$player_flag_1" == "random" ]]; then
                    heuristic_1="X"
                fi

                for ((l = 0; l < ${#H_FLAGS[@]}; l++)); do
                    if [ "$k" -ne "$l" ]; then
                        heuristic_2="${H_FLAGS[l]}"
                        if [[ "$player_flag_2" == "random" ]]; then
                            heuristic_2="X"
                        fi

                        COMBINATION_KEY="${player_flag_1}_${heuristic_1}_vs_${player_flag_2}_${heuristic_2}"
                        if [[ ! " ${UNIQUE_COMBINATIONS[@]} " =~ " ${COMBINATION_KEY} " ]]; then
                            UNIQUE_COMBINATIONS+=("$COMBINATION_KEY")
                        fi
                    fi
                done
            done
        fi
    done
done



# Total number of matches after deduplication
TOTAL_MATCHES=$((NUM_TESTS * ${#UNIQUE_COMBINATIONS[@]}))
CURRENT_MATCH=0
BAR_WIDTH=50

# Start executing matches
echo "Starting games..."
for ((i=1; i<=NUM_TESTS; i++)); do
    for COMBINATION_KEY in "${UNIQUE_COMBINATIONS[@]}"; do
        # Update progress bar
        ((CURRENT_MATCH++))
        PERCENT=$((CURRENT_MATCH * 100 / TOTAL_MATCHES))
        FILLED=$((PERCENT * BAR_WIDTH / 100))
        EMPTY=$((BAR_WIDTH - FILLED))

        # Extract player flags and heuristics from the combination key
        player_flag_1=$(echo "$COMBINATION_KEY" | cut -d'_' -f1)
        heuristic_1=$(echo "$COMBINATION_KEY" | cut -d'_' -f2)
        player_flag_2=$(echo "$COMBINATION_KEY" | cut -d'_' -f4)
        heuristic_2=$(echo "$COMBINATION_KEY" | cut -d'_' -f5)
        
        echo ""
        echo "Running match: ${player_flag_1}(${heuristic_1})_vs_${player_flag_2}(${heuristic_2})"
        # continue

        # REAL LOADING BAR WITHOUT DEBUG
        printf "\r[%-${BAR_WIDTH}s] %d%% (%d/%d)" \
            "$(printf '#%.0s' $(seq 1 $FILLED))" "$PERCENT" "$CURRENT_MATCH" "$TOTAL_MATCHES"
        
        # Paths for log files
        SERVER_LOG="$OUTPUTS_DIR/S_${player_flag_1}(${heuristic_1})_vs_${player_flag_2}(${heuristic_2}).log"
        WHITE_LOG="$OUTPUTS_DIR/W_${player_flag_1}(${heuristic_1})_vs_${player_flag_2}(${heuristic_2}).log"
        BLACK_LOG="$OUTPUTS_DIR/B_${player_flag_2}(${heuristic_2})_vs_${player_flag_1}(${heuristic_1}).log"

        # Create log files
        touch "$SERVER_LOG" "$WHITE_LOG" "$BLACK_LOG"

        # Run the server
        java -jar "Server.jar" > "$SERVER_LOG" 2>&1 &
        SERVER_PID=$!

        # Give time for the server to fully initialize
        sleep 1

        # Run the players
        python3 "$PLAYER_PATH/PLAYER.py" WHITE 60 127.0.0.1 "$player_flag_1" "$heuristic_1" > "$WHITE_LOG" 2>&1 &
        WHITE_PID=$!
        python3 "$PLAYER_PATH/PLAYER.py" BLACK 60 127.0.0.1 "$player_flag_2" "$heuristic_2" > "$BLACK_LOG" 2>&1 &
        BLACK_PID=$!

        # Wait for server and players to finish
        wait $SERVER_PID
        wait $WHITE_PID
        wait $BLACK_PID
    done
done

# New line after the progress bar
echo

# Delete non-game log files for parsing
echo "Deleting non-gamelogs..."
find "./logs" -type f ! -name "*_vs_*" -delete

# Parse the log files
echo "Start parsing log files:"
python3 extractor.py
