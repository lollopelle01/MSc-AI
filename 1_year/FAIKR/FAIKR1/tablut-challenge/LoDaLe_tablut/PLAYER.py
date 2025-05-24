import sys
import time
from socket_manager import SocketManager
from board import Board
from utils import *
from search import Node
import traceback
from heuristics import grey_heuristic, heuristic_1, heuristic_2, heuristic_3, INF
import random
import numpy as np

PORT = {"WHITE":5800, "BLACK":5801}

P_FLAGS = ["search", "random"]
H_FLAGS = ["grey", "flag-1", "flag-2", "flag-3"]

VERBOSE = True      # quickly enable/disable verbose
              
def main():
    
    ## Check that there aren't missing or excessing args ##
    if len(sys.argv) < 4:
        if VERBOSE : print("Wrong number of args, it should be: python3 __main__.py <color(White/Black)> <timeout(seconds)> <IP_server>")
        sys.exit(1)
    
    ## Check args are corrected ##
    # - Color
    color = sys.argv[1].upper()
    if color not in PORT.keys():
        if VERBOSE : print("Wrong color, it should be \"White\" or \"Black\"")
        sys.exit(1)
        
    # - Timeout
    try:
        timeout = int(sys.argv[2])
    except ValueError:
        if VERBOSE : print("Wrong timeout, it should be an integer")
        sys.exit(1)
    
    # - Ip address 
    ip = sys.argv[3]          
    if not check_ip(ip):
        if VERBOSE : print("Wrong ip format")
        sys.exit(1)
    port = PORT[color]
    
    ## Optimization args ####   TODO: togliere o assicurarsi funzioni anche se non passati
    
    # - Player FLAG
    FLAG = sys.argv[4] 
    if FLAG not in P_FLAGS:
        if VERBOSE : print(f"Wrong player_flag (must be in {P_FLAGS})")
        sys.exit(1) 
    
    # - Heuristic FLAG  
    if FLAG=="search":    
        h_flag = sys.argv[5]
        if h_flag not in H_FLAGS:
            if VERBOSE : print(f"Wrong heuristic_flag (must be in {H_FLAGS})")
            sys.exit(1)
        match h_flag:
            case "flag-1": heuristic = heuristic_1
            case "flag-2": heuristic = heuristic_2
            case "flag-3": heuristic = heuristic_3
            case "grey": heuristic = grey_heuristic
            case _ : raise ValueError(f"{h_flag} isn't an available heuristic")
    
    # # - Weights    # TODO: fare appena si trova euristica decente
    # if FLAG=="search":
    #     weights = sys.argv[6]
    #     if h_flag not in H_FLAGS:
    #         if VERBOSE : print(f"Wrong heuristic_flag (must be in {H_FLAGS})")
    #         sys.exit(1)
    
    ## Initialize the socket ##
    player_name = FLAG
    if player_name=="search":
        player_name=h_flag
    sock = SocketManager(ip, port, player_name) # ho tolto LoDaLe solo per il limite di caratteris
    sock.create_socket()
    sock.connect()
    
    ###############################################################################
    ### GAME CYCLE ################################################################
    ###############################################################################
    
    err_count = 0
    boards_history = []
    try:
        while True:
            initial_time = time.time()
            
            ## 1) Read current state / state updated by the opponent move
            try: current_state = sock.get_state()
            except TypeError: 
                if err_count==0 : 
                    err_count += 1
                    pass # if loses can't read
                else : raise Exception("Server error")
            board = Board(current_state)
            boards_history.append(board.board.copy())

            if VERBOSE : 
                print(f"Current table:\n")
                board.pretty_print()

            # If we are still playing
            match current_state["turn"] :
                case _ if current_state["turn"] == color:
                    if VERBOSE : print("It's your turn")

                    ## 2) Compute the move
                    match FLAG :
                        case "random":
                            n = Node(board)  
                            moves = [c.state[-1] for c in n.get_children(color)]
                            try:
                                move = random.sample(moves,1)[0]
                            except Exception:
                                if VERBOSE : print("Random: il root non ha figli")
                            
                        case "search" :
                            # Initialize current node
                            n = Node(board)  
                                                        
                            score, move = n.breadth_first(
                                maximizing_player=(color=="WHITE"),
                                heuristic=heuristic,
                                history=boards_history
                            )
                            
                            # consider initialization time + bonus time to conclude communication
                            timeout_move = timeout - (time.time()-initial_time) - 1

                            if (score,color)!=(INF-1, "WHITE") and (score,color)!=(-INF+1, "BLACK"):
                                score_ab, move_ab = n.minimax_alpha_beta(
                                    maximizing_player=(color=="WHITE"),
                                    depth=4,
                                    heuristic=heuristic, 
                                    history=boards_history,
                                    timeout=timeout_move,
                                    start_time=time.time()
                                )
                                if type(score_ab)!=str:
                                    move = move_ab       
                                    score = score_ab                         
                                else:
                                    if VERBOSE : print("Timeout scattato")

                            if VERBOSE : 
                                print("\nExploration recap:")
                                # n.plot_tree(heuristic=heuristic)
                                print(f"- Score={score}")
                                print(f"- Nodes_explored={n.get_num_nodes("explored")}")
                                print(f"- Time={round(time.time()-initial_time, 3)}s")
                                print()
                                
                    if VERBOSE : print(move)

                    ## 3) Send the move
                    sock.send_move(move.to_alfanum_tuple())
                    
                    ## 4) Read the new updated state after my move
                    current_state = sock.get_state()
                    board = Board(current_state) 
                    boards_history.append(board.board.copy())
                    
                    if VERBOSE : 
                        print(f"After my move:\n")
                        board.pretty_print()
                        
                case "WHITEWIN":
                    if VERBOSE : print("WHITE wins!")
                    sys.exit(0)
                    
                case "BLACKWIN":
                    if VERBOSE : print("BLACK wins!")
                    sys.exit(0)
                    
                case "DRAW":
                    if VERBOSE : print("It's a draw!")
                    sys.exit(0)
                    
                case _ :
                    if VERBOSE : print("Waiting for your opponent move...")

    except Exception:
        if VERBOSE : print(f"An error occurred during the game:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()