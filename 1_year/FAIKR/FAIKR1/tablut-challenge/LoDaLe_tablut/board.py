import numpy as np
from utils import *

# CONSTANTS
WHITE = "WHITE"
BLACK = "BLACK"
KING = "KING"
EMPTY = "EMPTY"
CAMP = "CAMP"
ESCAPE = "S-ESCAPE"

CASTLE = (4, 4)

CAMPS = {   
    "up" : [(0,3), (0,4), (0,5), (1,4)],     # upper camps
    "down" : [(8,3), (8,4), (8,5), (7,4)],     # lower camps
    "left" : [(3,0), (4,0), (5,0), (4,1)],     # left camps
    "right" : [(3,8), (4,8), (5,8), (4,7)]      # right camps
}
CAMPS["not-border"] = [camp[3] for camp in CAMPS.values()]
CAMPS["all"] = CAMPS["up"] + CAMPS["down"] + CAMPS["left"] + CAMPS["right"]

ESCAPES = {   
    "up" : [(1,0), (2,0), (6,0), (7,0)],
    "right" : [(8,1), (8,2), (8,6), (8,7)],
    "down" : [(1,8),(2,8),(6,8),(7,8)],
    "left" : [(0,1),(0,2),(0,6),(0,7)]
}
ESCAPES["all"] = ESCAPES["up"] + ESCAPES["right"] + ESCAPES["down"] + ESCAPES["left"] 

ROMBUS_POS = [
            (1,2),       (1,6),
    (2,1),                      (2,7),

    (6,1),                      (6,7),
            (7,2),       (7,6)
]
ROMBUS_POS = np.array(ROMBUS_POS)

ESCAPE_COVER = [
    (1,1),(1,2),        (1,6),(1,7),
    (2,1),                    (2,7),

    (6,1),                    (6,7),
    (7,1),(7,2),        (7,6),(7,7)
]
ESCAPE_COVER = np.array(ESCAPE_COVER)

DOUBLE_ESCAPE_COVER = [
    (1,1),          (1,7),

    (7,1),          (7,7)
]
DOUBLE_ESCAPE_COVER = np.array(DOUBLE_ESCAPE_COVER)

##
## Move class incapsulates a single move in the Board
##

class Move:
    def __init__(self, start:tuple, end:tuple, piece, is_capture = False):
        self.is_capture = is_capture
        if is_capture:
            self.start = start
            self.end = self.start
            self.piece = piece
            return
        if self.is_within_bounds(end):
            self.start = start
            self.end = end
        else:
            raise Exception(f"'Move' class detected non valid x,y position : {start} --> {end}")
        if not self.is_orthogonal():
            raise Exception("'Move' class detected non orthogonal move")
        if piece == WHITE or piece == BLACK or piece == KING:
            self.piece = piece
        else:
            raise Exception("'Move' class detected non valid piece")
        
    def is_within_bounds(self, pos):
        return 0 <= pos[0] < 9 and 0 <= pos[1] < 9
    
    def is_orthogonal(self):
        return (self.start[0] == self.end[0]) or (self.start[1] == self.end[1])
    
    # returns a tuple in the format of (from_alfanum, to_alfanum)
    def to_alfanum_tuple(self) -> tuple:
        col_start, row_start  = self.start
        col_end, row_end = self.end
        alfnum_start = chr(ord('a') + col_start)
        alfnum_start += str(1+row_start)
        alfnum_end = chr(ord('a') + col_end)
        alfnum_end += str(1+row_end)
        if self.piece == WHITE or self.piece == KING:
            color = "WHITE"
        else:
            color = "BLACK"
        return (alfnum_start, alfnum_end, color)
    
    def __eq__(self, other):
        return (self.start == other.start) and (self.end == other.end) and (self.piece == other.piece)
    
    def __str__(self):
        start, end, _ = self.to_alfanum_tuple() 
        return f'Move {self.piece} : {start} --> {end}'
    
##
## Board class serves as a domain representation for the Tablut game. It can operate on the grid following the
## the rules described in the Tablut Challenge pdf
##

class Board:

    def __init__(self, state):
        self.turn = state['turn']
        self.board = np.transpose(np.array(state['board']))
        king_pos = np.where(self.board == KING)
        self.king_captured = False
        if king_pos[0].size == 0:
            self.king = None
        elif king_pos[0].size == 1:
            self.king = (king_pos[0][0], king_pos[1][0])
        else:
            raise Exception("More than one king in the board") 
    
    # controlla se la data posizione è interna alla griglia  ?? da togliere perchè aggiunto come metodo statico di classe Move
    def is_within_bounds(self, pos):
        return 0 <= pos[0] < 9 and 0 <= pos[1] < 9

    # controlla se due pezzi sono di fazione diversa
    def is_opponent_piece(self, piece_one, piece_two):
        if (piece_one == WHITE and piece_two == KING) or (piece_one == KING and piece_two == WHITE):
            return False
        if piece_one != piece_two:
            return True
        return False
    
    def get_cell(self, pos): 
        if self.is_within_bounds(pos):
            cell = self.board[pos[0]][pos[1]]
            if cell == EMPTY:
                if pos in CAMPS["all"]:
                    cell = CAMP
                elif pos in ESCAPES["all"]:
                    cell = ESCAPE
            return cell
        else:
            return "-"
    
    # given a segment of cells identified by starting position pos1 and end position pos2
    # returns a string which gives a summary of the content of each cell ordered from pos1 to pos2
    # in the segment (starting point not included, end point included), format example: 'bbwee...' there are two blacks, one white and two empty.
    # A cell which is not in bound is flagged in the string with '-' char 
    def segment_occupation(self, pos1, pos2) -> dict:
        str_result = ""
        cells_result = []
        delta = (pos2[0]-pos1[0],pos2[1]-pos1[1])
        # when the move is vertical
        if delta[0] == 0:
            sign = np.sign(delta[1])
            step = sign
            if step == 0:
                step = 1
            for i in range( sign, delta[1]+sign, step ):
                # get the frist letter of each cell in the path of the segment B for black W for white E for empty K for king
                letter = self.get_cell((pos1[0], pos1[1]+i))[0]
                str_result += letter
                cells_result.append( (pos1[0], pos1[1]+i) )
        # when the move is orizontal
        elif delta[1] == 0:
            sign = np.sign(delta[0])
            step = sign
            if step == 0:
                step = 1
            for i in range( sign, delta[0]+sign, step ):
                # get the frist letter of each cell in the path of the segment B for black W for white E for empty K for king
                letter = self.get_cell((pos1[0]+i, pos1[1]))[0]
                str_result += letter
                cells_result.append( (pos1[0]+i, pos1[1]) )
        return {"str":str_result, "cells":cells_result}
    
    # given a center in the grid, it returns the Summary Object for the cells in the ring (of radius 1 or 2)
    # the Summary Object is a dictionary which has under the key "str" the string containing the info about cell occupations,
    # under the key "cells" the position tuple corresponding to the string letter.
    # A cell which is not in bound is flagged in the string with '-' char 
    def ring_occupation(self, center, radius) -> dict:
        str_result = ""
        cell_result = []
        if radius < 1 or radius > 2:
            raise Exception("In 'ring_occupation' radius must be either 1 or 2")
        # take the corners of the ring around the center (in order: top-left, top-right, bottom-right, bottom-left)
        corners = [(center[0]-radius, center[1]-radius),(center[0]+radius, center[1]-radius),(center[0]+radius, center[1]+radius),(center[0]-radius, center[1]+radius)]
        # get the summary object for the content of the ring, using the method 'segment_occupation'
        # the resulting object is composed by the cells of the content of cells starting from top left corner (not included) going in 
        # clockwise direction (until it includes the top left corner)
        for i in range(4):
            occupation_obj = self.segment_occupation(corners[i%4], corners[(i+1)%4])
            str_result += occupation_obj["str"]
            for cell in occupation_obj["cells"]:
                cell_result.append( cell )
        return {"str": str_result, "cells": cell_result}
    
    # check whether pos1 and pos2 positions are adjacent on the grid (they have a common side)
    def is_adjacent(self, pos1, pos2):
        delta = (pos1[0]-pos2[0], pos1[1]-pos2[1])
        return (delta[0] == 0 and abs(delta[1]) == 1) or (delta[1] == 0 and abs(delta[0]) == 1)

    def is_valid_move(self, move:Move):
        # The starting position must contain the piece specified in move
        if self.get_cell(move.start) != move.piece:
            return False
        # only the king can move in the castle
        if move.end == CASTLE and move.piece != KING:
            return False
        # only the black can move in camps
        if move.end in CAMPS['all'] and move.piece != BLACK:
            return False
        # if a black piece is outside a camp it cannot re-enter
        if (move.end in CAMPS['all']) and (move.start not in CAMPS['all']):
            return False
        return True
    
    def is_king_escaped(self):
        return self.king in ESCAPES["all"]
    
    def is_king_captured(self):
        return self.king_captured
    
    # returns a tuple (True, pos) if this move brings to the capture of a piece in the 'pos' position
    # else it returns (False, xx). IT DOESN'T CHECK WHETHER IS A VALID MOVE (you should check before calling this method)
    def is_a_capture_move(self, move: Move):
        # if the moving piece is black, check if it is capturing the king
        if move.piece == BLACK:
            # if the end position of the move has a king adjacent
            if self.is_adjacent(move.end, self.king):
                # check the surrounding of the king in even positions (up, down, left, right)
                king_surround = self.ring_occupation(self.king, 1)
                black_occupation = find_all(king_surround["str"], BLACK[0])
                blacks_in_even_position = 0
                for index in black_occupation:
                    if index % 2 == 0:
                        blacks_in_even_position += 1
                # If the King is in the Castle, it must be already surrounded on 3 sides (the 4th is the end position of this move)
                if self.king == CASTLE:
                    return (blacks_in_even_position==3, self.king)
                # if the king is adjacent to the castle, it must be already surrounded on 2 sides (the 3rd is the end position of this move)
                if self.is_adjacent(self.king, CASTLE):
                    return (blacks_in_even_position==2, self.king)
                # if the end position makes the king between a black piece and a camp, it is a capture
                # check for every even surrounding cell if the king has on opposite sides the end of this move and a camp
                for cell in king_surround["cells"]:
                    if cell in CAMPS["all"]:
                        delta = (cell[0]-move.end[0], cell[1]-move.end[1])
                        if (delta[0] == 0 and abs(delta[1]) == 2) or (delta[1] == 0 and abs(delta[0]) == 2):
                            return (True, self.king)
        # now all the general rules for all checkers BLACK, WHITE and KING
        surround = self.ring_occupation(move.end, 1) # take the surround of the move end point
        string_indexes = []
        if move.piece == WHITE or move.piece == KING: # if the moved piece is white or king we are interested in black 
            string_indexes = find_all(surround["str"], BLACK[0])
        if move.piece == BLACK: # if the moved piece is black we are interested in white and king
            string_indexes = find_all(surround["str"], WHITE[0]) + find_all(surround["str"], KING[0])
        # for each surrounding adversary
        for elem in string_indexes:
            if elem % 2 == 0:
                # at this point elem is an adjacent adversary 1.1 0.1  1.0
                delta = (move.end[0] - surround["cells"][elem][0], move.end[1] - surround["cells"][elem][1])
                opposite_cell_coordinates = (surround["cells"][elem][0] - delta[0], surround["cells"][elem][1] - delta[1])
                if self.get_cell(opposite_cell_coordinates) == move.piece: # it means that the cell of the adversary is between an ally and your next move, so this move is a capture
                    return (True, (surround["cells"][elem][0], surround["cells"][elem][1]))
                # check if the opposite cell is the castle, or a camp. In this case this move is a capture
                if opposite_cell_coordinates == CASTLE or opposite_cell_coordinates in CAMPS["all"]:
                    return (True, (surround["cells"][elem][0], surround["cells"][elem][1]))
        return (False, None)
        
    # take a position on the grid and returns the list of all possible legal moves for that piece
    def get_all_moves_for_piece(self, pos):
        if not self.is_within_bounds(pos):
            raise Exception("Position not in the grid for 'get_all_moves_for_piece' call")
        piece = self.get_cell(pos)
        is_in_camp = piece == BLACK and pos in CAMPS["all"]
        if piece == EMPTY or piece == CAMP or piece == ESCAPE:
            raise Exception("Empty piece when calling 'get_all_moves_for_piece'")
        # coordinates of all the cells on the border of the grid moving orthogonally
        borders = [(pos[0], 0), (8, pos[1]), (pos[0], 8), (0, pos[1])]
        segment_occupations = [self.segment_occupation(pos, borders[0]), 
                               self.segment_occupation(pos, borders[1]),
                               self.segment_occupation(pos, borders[2]),
                               self.segment_occupation(pos, borders[3])]
        result = []
        for direction in segment_occupations:
            for i in range(len(direction["str"])):
                if (piece == BLACK and is_in_camp) and (direction["str"][i] != EMPTY[0] and \
                                                        direction["str"][i] != ESCAPE[0] and \
                                                        direction["str"][i] != CAMP[0]  ):
                    break
                elif not (piece == BLACK and is_in_camp) and (direction["str"][i] != EMPTY[0] and direction["str"][i] != ESCAPE[0]):
                    break
                move = Move(pos, direction["cells"][i], piece)
                if self.is_valid_move(move):
                    result.append(move)
        return result
    
    # it calculates all the free ways towards all the escapes, and it returns a list with all the cells part of those escape ways
    def get_highlighted_escape_cells(self):
        # distinguish the computation for each side of escapes (up, right, bottom, left)
        directions = [(0,1),(-1,0),(0,-1),(1,0)]
        result = []
        # pair a direction vector for each side of ESCAPES
        for vector, escapes in zip(directions, ESCAPES):
            escapes = ESCAPES[escapes] # get escapes list of that side
            for escape in escapes: # for each escape check the stright direction (according to the vector direction)
                check_pos = (escape[0] + vector[0], escape[1] + vector[1]) # start from that escape
                # check until you reach a black or a camp. For a king it's important to highlight the cells even if they trespass a WHITE
                while self.get_cell(check_pos) != BLACK and self.get_cell(check_pos) != CAMP and self.is_within_bounds(check_pos):
                    result.append(check_pos) # add that position
                    check_pos = (check_pos[0] + vector[0], check_pos[1] + vector[1]) # move toward vector direction
        return result


    
    # given a list of Move, it applies them in order (the moves are supposed to be altready legal)
    def apply_moves(self, moves_list:list):
        sequence_list = []
        for move in moves_list:
            self.board[move.start[0]][move.start[1]] = EMPTY
            self.board[move.end[0]][move.end[1]] = move.piece
            if move.piece == KING:
                self.king = move.end
            sequence_list.append(move)
            # check if there is a capture to perform
            capturing_result = self.is_a_capture_move(move)
            if capturing_result[0]:
                captured_piece = self.board[capturing_result[1][0]][capturing_result[1][1]]
                self.board[capturing_result[1][0]][capturing_result[1][1]] = EMPTY
                # add the capture action to the sequence list so it can be tracked down and eventually reversed
                sequence_list.append(Move((capturing_result[1][0], capturing_result[1][1]), 0, captured_piece, True))
                if captured_piece == KING:
                    self.king_captured = True
        return sequence_list
    
    # given a list of Move, it applies them in reverse order and from end to start (the moves are supposed to be altready legal)
    def reverse_moves(self, sequence_list:list,):
        for move in sequence_list[::-1]:
            if move.is_capture:
                self.board[move.start[0]][move.start[1]] = move.piece
                if move.piece == KING:
                    self.king = (move.start[0], move.start[1])
            else:
                self.board[move.end[0]][move.end[1]] = EMPTY
                self.board[move.start[0]][move.start[1]] = move.piece
                if move.piece == KING:
                    self.king = move.start
    
    def get_all_pieces_of_color(self, color):
        mask = (self.board == color)
        positions = np.where(mask)
        position_list = []
        for i in range(positions[0].size):
            position_list.append((positions[0][i], positions[1][i]))
        return position_list
    
    # returns all the current possible moves in the board, based on the specified color
    def get_all_moves(self, color, moves):
        result = []
        move_sequence = self.apply_moves(moves)
        if color == WHITE:
            positions = self.get_all_pieces_of_color(WHITE) + self.get_all_pieces_of_color(KING)
        else:
            positions = self.get_all_pieces_of_color(BLACK)
        for i in range(len(positions)):
            moves = self.get_all_moves_for_piece(positions[i])
            for mv in moves:
                result.append(mv)
        self.reverse_moves(move_sequence)
        return result
    
    def get_king(self):
        return self.king

    def pretty_print(self):
        board = np.transpose(self.board)
        n = board.shape[0]
        column_labels = '    ' + '   '.join([chr(i + ord('A')) for i in range(n)])  # Quattro spazi
        print(column_labels)
        separator = '  ' + '+---' * n + '+'
        for i in range(n):
            print(separator) 
            # Row + pieces
            row = f'{i+1:2d} ' + '|'.join([f' {cell[0]} ' if cell != 'EMPTY' else '   ' for cell in board[i]]) + ' |'
            print(row)
        print(separator)