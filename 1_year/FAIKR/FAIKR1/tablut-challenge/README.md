# LoDaLe Group

LoDaLe is a Python client for the Ashton Tablut game, developed for the Tablut Challenge at the University of Bologna.

## Project Structure

```
.
├── LoDaLe_tablut
│   ├── PLAYER.py          # Main player logic and game loop
│   ├── board.py           # Board representation and utility functions
│   ├── heuristics.py      # Heuristic functions for evaluating board states
│   ├── search.py          # Search algorithms like Breadth-first and Alpha-Beta Pruning
│   ├── socket_manager.py  # Socket communication with the game server
│   └── utils.py           # General utility functions
|
├── Test
│   ├── Server.jar         # Game server for testing the client
│   └── logs               # Logs generated during gameplay
|
├── dataset                # Parsed dataset given from previous years
│   ├── ...
├── games_executor         # Application for comparing different players
│   ├── ...
|
├── run_game.sh            # Script for launching the game in Unix
├── run_game.bat           # Script for launching the game in Windows
```

## Launching

In order to launch the player type 
```
python3 PLAYER.py <color_player> <timeout> <server_ip>
```

## Observations

We did not have enough time to implement and test other ideas to improve the player, such as:
- **Genetic algorithm** to refine the weights of the heuristics
- **Iterative deepening** with **transposition table** over the alpha-beta cut, in order to guarantee a solution with respect to the available time.

---
Developed with ❤️ by [Lorenzo Pellegrino](https://github.com/lollopelle01), [Leonardo Massaro](https://github.com/leomass), and [Davide Ligabue](https://github.com/davideligabue).
