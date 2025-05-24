import ipaddress
from collections import Counter
import time


def check_ip(ip):
    '''
    Checks if the IP address is well-formatted
    '''
    try:
        ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False
    
def tuple2alfanum(tuple_pos):
    row, col = tuple_pos
    col_alfnum = chr(ord('a') + col)
    row_alfnum = str(1+row)
    return col_alfnum + row_alfnum

def alfnum2tuple(alfnum_pos):
    col_alfnum, row_alfnum = tuple(alfnum_pos)
    col_tuple = ord(col_alfnum) - ord('a')
    row_tuple = int(row_alfnum) - 1
    return (row_tuple, col_tuple)

# take a string and a char and returns a list of all the indexes in which the char is found
def find_all(original:str, tofind:str) -> list:
    result = list()
    for i in range(len(original)):
        if original[i] == tofind:
            result.append(i)
    return result

def get_n_most_winning_move(dataset, results, n, color):

    # Filter idxs
    max_turns = max([len(match) for match in dataset])
    if color[0]=="W" :
        possible_idx = [2*i for i in range(max_turns)]   
    else :
        possible_idx = [2*i+1 for i in range(max_turns)]
    
    idx = possible_idx[n]          # which move (column)

    idx_moves = []
    for i, match in enumerate(dataset) :
        try:
            if results[i]==color[0] : # if wins
                idx_moves.append(tuple(match[idx]))
        except: continue
        
    move_counter = Counter(idx_moves)
    most_common_moves = move_counter.most_common(None) # all the results
    return most_common_moves

def is_timeout(start_time, timeout, bonus=1) -> bool:
    res = time.time() - start_time > timeout-bonus
    # print(f"Timeout: ({time.time()} - {start_time} > {timeout} - {bonus}) ==== {res}")
    return res
