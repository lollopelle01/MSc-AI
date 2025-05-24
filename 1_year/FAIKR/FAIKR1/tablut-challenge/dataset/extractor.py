import numpy as np
import re
import os
from tqdm import tqdm

# def parse(board_list):
#     mapping = {
#         'O': 'EMPTY',
#         'B': 'BLACK',
#         'W': 'WHITE',
#         'K': 'KING',
#         'T': 'TOWER'
#     }
#     parsed_board = np.array([[mapping[char] for char in row] for row in board_list], dtype='<U5')
#     return parsed_board

if __name__ == "__main__":
    path = "dataset/games"
    files = os.listdir(path)
    files.sort()
    
    # FATTA MALE
    # result_tables = [[] for _ in files]
    # # Iterate over each file in the specified directory with tqdm for file loading progress bar
    # for i, filename in enumerate(tqdm(files, desc="Loading files")):
    #     file_path = os.path.join(path, filename)
    #     if os.path.isfile(file_path):
    #         with open(file_path, "r") as file:
    #             content = file.read()
    #             tables_united = [line[-9:] for line in content.splitlines() if re.search(r'[OBWKT]{9}', line)]
    #             tables_divided = [tables_united[i:i+9] for i in range(0, len(tables_united), 9)]
                
    #             for table in tables_divided:
    #                 parsed_table = parse(table)
    #                 result_tables[i].append(parsed_table)
    
    # # for i, match in enumerate(result_tables) :
    # #     print(f"{files[i]} \t {len(match)} moves")
        
    
    result_moves = []
    result_games = []
    for i, filename in enumerate(tqdm(files, desc="Loading files")):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path):
            pattern = re.compile(r"from\s+([a-iA-I][1-9])\s+to\s+([a-iA-I][1-9])", re.IGNORECASE)
            with open(file_path, "r") as file:
                
                # Memorize the moves
                lines = file.readlines()
                filtered_lines = [line for line in lines if pattern.search(line)]
                moves = [
                            (line.strip().lower()[-8:-6],   # from
                             line.strip().lower()[-2:])     # to
                            
                        for line in filtered_lines]
                if moves != [] : # null moves to be discarded
                    result_moves.append(np.array(moves))
                    
                    # Check result of match
                    last_non_empty_line = next((line.strip() for line in reversed(lines) if line.strip()), None)
                    if last_non_empty_line.strip().endswith("WW") : result_games.append("W")        # white wins
                    elif last_non_empty_line.strip().endswith("BW") : result_games.append("B")      # black wins
                    elif last_non_empty_line.strip().endswith("D") : result_games.append("D")       # draw
                    else : 
                        result_games.append("I")                                                 # interrupted / bad formatted
                        print(file_path)
                    
    result_moves = np.array(result_moves, dtype=object)
    result_games = np.array(result_games, dtype=str)
                
    # for i, match in enumerate(result_moves) :
    #     print(f"{files[i]} \t {len(match)} moves")
    
    # Optionally save the results
    # np.save("dataset_tabelle.npy", result_tables)
    np.save("dataset/dataset_moves.npy", result_moves)
    np.save("dataset/dataset_results.npy", result_games)
