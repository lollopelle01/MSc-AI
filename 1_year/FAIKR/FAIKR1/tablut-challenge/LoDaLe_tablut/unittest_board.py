from board import *
import numpy as np
import unittest
from search import Node


'''
Tests for class Board
'''


class TestBoard(unittest.TestCase):

    def initialize_Board(self) -> Board:
        state = {
            'turn': 'WHITE',
            'board': np.array([
                ['EMPTY', 'EMPTY', 'EMPTY', 'BLACK', 'BLACK', 'BLACK', 'EMPTY', 'EMPTY', 'EMPTY'],
                ['EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'BLACK', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY'],
                ['EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'WHITE', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY'],
                ['BLACK', 'EMPTY', 'EMPTY', 'EMPTY', 'WHITE', 'EMPTY', 'EMPTY', 'EMPTY', 'BLACK'],
                ['BLACK', 'BLACK', 'WHITE', 'WHITE', 'KING', 'WHITE', 'WHITE', 'BLACK', 'BLACK'],
                ['BLACK', 'EMPTY', 'EMPTY', 'EMPTY', 'WHITE', 'EMPTY', 'EMPTY', 'EMPTY', 'BLACK'],
                ['EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'WHITE', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY'],
                ['EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'BLACK', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY'],
                ['EMPTY', 'EMPTY', 'EMPTY', 'BLACK', 'BLACK', 'BLACK', 'EMPTY', 'EMPTY', 'EMPTY']
            ])
        }
        return Board(state)
    
    def initialize_Board_simple_case(self) -> Board:
        state = {
            'turn': 'WHITE',
            'board': np.array([
                ['EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY'],
                ['EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY'],
                ['EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'BLACK', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY'],
                ['EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'WHITE', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY'],
                ['EMPTY', 'EMPTY', 'KING', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY'],
                ['BLACK', 'EMPTY', 'EMPTY', 'EMPTY', 'WHITE', 'EMPTY', 'EMPTY', 'EMPTY', 'BLACK'],
                ['EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY'],
                ['EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY'],
                ['EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY']
            ])
        } # 8+6+12+11+9+11
        return Board(state)# BLACK ... | WHITE ..

    def initialize_Board_third_case(self) -> Board:
        state = {
            'turn': 'WHITE',
            'board': np.array([
                ['EMPTY', 'EMPTY', 'EMPTY', 'BLACK', 'BLACK', 'BLACK', 'EMPTY', 'EMPTY', 'EMPTY'],
                ['EMPTY', 'EMPTY', 'BLACK', 'EMPTY', 'EMPTY', 'EMPTY', 'BLACK', 'BLACK', 'EMPTY'],
                ['EMPTY', 'BLACK', 'EMPTY', 'EMPTY', 'WHITE', 'EMPTY', 'EMPTY', 'BLACK', 'EMPTY'],
                ['BLACK', 'EMPTY', 'EMPTY', 'EMPTY', 'WHITE', 'EMPTY', 'EMPTY', 'EMPTY', 'BLACK'],
                ['BLACK', 'EMPTY', 'WHITE', 'WHITE', 'KING', 'WHITE', 'WHITE', 'BLACK', 'BLACK'],
                ['BLACK', 'EMPTY', 'BLACK', 'EMPTY', 'WHITE', 'EMPTY', 'EMPTY', 'EMPTY', 'BLACK'],
                ['EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'WHITE', 'EMPTY', 'BLACK', 'EMPTY', 'EMPTY'],
                ['EMPTY', 'BLACK', 'EMPTY', 'EMPTY', 'BLACK', 'EMPTY', 'BLACK', 'BLACK', 'EMPTY'],
                ['EMPTY', 'EMPTY', 'EMPTY', 'BLACK', 'BLACK', 'BLACK', 'EMPTY', 'EMPTY', 'EMPTY']
            ])
        } 
        return Board(state)
    
    def initialize_Board_4_case(self) -> Board:
        state = {
            'turn': 'WHITE',
            'board': np.array([
                ['EMPTY', 'WHITE', 'EMPTY', 'EMPTY', 'BLACK', 'BLACK', 'EMPTY', 'EMPTY', 'EMPTY'],
                ['EMPTY', 'BLACK', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'BLACK', 'WHITE'],
                ['BLACK', 'BLACK', 'BLACK', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'BLACK', 'EMPTY'],
                ['EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'BLACK', 'EMPTY', 'EMPTY', 'EMPTY', 'BLACK'],
                ['BLACK', 'EMPTY', 'EMPTY', 'BLACK', 'KING', 'EMPTY', 'WHITE', 'EMPTY', 'BLACK'],
                ['EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY'],
                ['EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY'],
                ['EMPTY', 'BLACK', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'BLACK', 'EMPTY', 'BLACK'],
                ['EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'BLACK', 'BLACK', 'EMPTY', 'EMPTY', 'EMPTY']
            ]) # 1,2 black -> 1,6 blsck
        } 
        return Board(state)
    
    def initialize_Board_5_case(self) -> Board:
        state = {
            'turn': 'WHITE',
            'board': np.array([
                ['EMPTY', 'WHITE', 'EMPTY', 'EMPTY', 'BLACK', 'BLACK', 'EMPTY', 'EMPTY', 'EMPTY'],
                ['EMPTY', 'BLACK', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'BLACK', 'WHITE'],
                ['BLACK', 'BLACK', 'BLACK', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'BLACK', 'EMPTY'],
                ['EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'BLACK', 'EMPTY', 'EMPTY', 'EMPTY', 'BLACK'],
                ['BLACK', 'EMPTY', 'EMPTY', 'BLACK', 'KING', 'EMPTY', 'WHITE', 'EMPTY', 'BLACK'],
                ['EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY'],
                ['EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY'],
                ['EMPTY', 'BLACK', 'EMPTY', 'BLACK', 'EMPTY', 'EMPTY', 'BLACK', 'EMPTY', 'BLACK'],
                ['EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'BLACK', 'BLACK', 'EMPTY', 'EMPTY', 'EMPTY']
            ]) # 1,2 black -> 1,6 blsck
        } 
        return Board(state)
    
    # def test_board_initialization(self):
    #     board = self.initialize_Board()
    #     self.assertEqual(board.get_cell((3,0)), "BLACK")
    #     self.assertEqual(board.get_cell((4,4)), "KING")
    #     self.assertEqual(board.get_cell((6,4)), "WHITE")

    #     board = self.initialize_Board_simple_case()
    #     self.assertEqual(board.get_cell((2,4)), "KING")
    #     self.assertEqual(board.get_cell((4,5)), "WHITE")
    #     self.assertEqual(board.get_cell((0,5)), "BLACK")

    # def test_king_position(self):
    #     board = self.initialize_Board()
    #     self.assertEqual(board.king, CASTLE, "Bad king initializazion in starting position")
    #     board = self.initialize_Board_simple_case()
    #     self.assertEqual(board.king, (2,4), "Bad king initialization in different position")

    # def test_performing_checks(self):
    #     board = self.initialize_Board()
    #     # is_orthogonal and is_within_bounds check
    #     self.assertRaises(Exception, lambda: Move((0,3),(12,3),BLACK), "Should throw exception because out of bound move")
    #     self.assertRaises(Exception, lambda: Move((0,3),(1,2),BLACK), "Should throw exception because not orthogonal")
    #     self.assertRaises(Exception, lambda: Move((0,3),(1,3),"wrong piece"), "Should throw exception because 'piece' is incorrect")
    #     self.assertIsInstance(Move((0,3),(2,3),BLACK), Move, "A correct Move initialization")
    #     # get_cell check
    #     self.assertRaises(Exception, lambda: board.get_cell((9,7)), "Should throw exception because checks a cell out of bounds")
    #     self.assertEqual(board.get_cell((3,8)), BLACK, "Incorrect value returned by get_cell")
    #     # is_adjacent check
    #     self.assertTrue(board.is_adjacent((5,6),(4,6)), "is_adjacent should have returned True")
    #     self.assertFalse(board.is_adjacent((5,6),(4,7)), "is_adjacent should have returned False")
    #     # is_valid_move check
    #     board = self.initialize_Board_simple_case()
    #     self.assertTrue(board.is_valid_move(Move((0,5),(0,3),BLACK)), "Black piece must be able to move in camps if already inside")
    #     self.assertTrue(board.is_valid_move(Move(board.king,CASTLE,KING)), "King should be able to move in CASTLE")
    #     self.assertFalse(board.is_valid_move(Move((4,2),(4,1),BLACK)), "Black can't move in camps it was outside")
    #     self.assertFalse(board.is_valid_move(Move((4,5),(3,5),BLACK)), "If move.piece is Black you cannot move a white piece")
    #     self.assertFalse(board.is_valid_move(Move((4,5),CASTLE,WHITE)), "Only king should be able to move in CASTLE")

    # def test_initial_ring_occupation(self):
    #     board = self.initialize_Board()
    #     occupation_obj = {
    #         "str":"WEWEWEWE",
    #         "cells" : [(4,3),(5,3),(5,4),(5,5),(4,5),(3,5),(3,4),(3,3)]
    #     }
    #     self.assertDictEqual(occupation_obj, board.ring_occupation((4,4), 1), "Error object ring occupation")
    #     occupation_str = "EEEWWKWWEEESBBBS"
    #     self.assertEqual(occupation_str, board.ring_occupation((2,4), 2)["str"], "Error string of ring (n=2) occupation")

    # def test_move_pieces_around(self):
    #     board = self.initialize_Board_simple_case()
    #     # moving pieces
    #     move_seq = board.apply_moves([Move((4,5),(1,5),WHITE), Move((8,5),(2,5),BLACK)])
    #     self.assertEqual(len(move_seq), 3, "Wrong sequence of moves returned by apply_moves when capturing a checker")
    #     new_state = np.transpose( np.array([
    #             ['EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY'],
    #             ['EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY'],
    #             ['EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'BLACK', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY'],
    #             ['EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'WHITE', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY'],
    #             ['EMPTY', 'EMPTY', 'KING', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY'],
    #             ['BLACK', 'EMPTY', 'BLACK', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY'],
    #             ['EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY'],
    #             ['EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY'],
    #             ['EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY']
    #         ]) )
    #     self.assertTrue(np.array_equiv(new_state, board.board), "Wrong expected state after moves and checker capture")
    #     # reversing the moves
    #     old_state = self.initialize_Board_simple_case().board
    #     board.reverse_moves(move_seq)
    #     self.assertTrue(np.array_equiv(old_state, board.board), "Wrong expected state after reversing moves")
    #     # capturing king
    #     # KING 2,4 -> 2,5 | WHITE 4,5 -> 4,6 | BLACK 0,5 -> 1,5 | BLACK 8,5 -> 3,5
    #     move_seq = board.apply_moves([Move((2,4),(2,5),KING), Move((4,5),(4,6),WHITE), Move((0,5),(1,5),BLACK), Move((8,5),(3,5),BLACK)])
    #     self.assertTrue(board.is_king_captured(), "King should be captured")
    #     self.assertFalse(board.is_king_escaped(), "King should not be escaped")
    #     board.reverse_moves(move_seq)
    #     # escape king
    #     board.apply_moves([Move((2,4),(2,0),KING)])
    #     self.assertTrue(board.is_king_escaped(), "King should be escaped")
    #     # trespass a camp for a black piece
    #     board = self.initialize_Board_4_case()
    #     self.assertSequenceEqual([Move((1,2), (1,3), BLACK)], board.get_all_moves_for_piece((1,2)), "Black should not move through a camp")

    # def test_get_global_possibilities(self):
    #     board = self.initialize_Board_simple_case()
    #     # test get all pieces
    #     self.assertTrue(len(board.get_all_pieces_of_color(WHITE)) == 2, "Wrong matching number of white pieces")
    #     self.assertTrue(len(board.get_all_pieces_of_color(BLACK)) == 3, "Wrong matching number of white pieces")
    #     self.assertTrue(len(board.get_all_pieces_of_color(KING)) == 1, "Wrong king found")
    #     # test get all moves
    #     self.assertTrue(len(board.get_all_moves(KING, [])) == 12, "Wrong number of possible moves for king")
    #     self.assertTrue(len(board.get_all_moves(KING, [])) + len(board.get_all_moves(BLACK, [])) + len(board.get_all_moves(WHITE, [])) == 45, "Wrong number of possible total moves")

    # def test_highlighted_cells(self):
    #     board = self.initialize_Board_third_case()
    #     # 1,1  7,6   2,7  2,6  1,6  2,6  3,6  4,6  5,6  1,1
    #     position_set_target = set([(1,1), (7,6), (2,7), (2,6), (1,6), (2,6), (3,6), (4,6), (5,6), (1,1)])
    #     self.assertEqual(set(board.get_highlighted_escape_cells()), position_set_target, "Wrong highlighted escape cells")
        
    def isin_test(self):
        board_history=[]
        
        b1=self.initialize_Board()
        b2=self.initialize_Board_simple_case()
        b3=self.initialize_Board_4_case()
        b4=self.initialize_Board_third_case()
        b5=self.initialize_Board_5_case()
        
        board_history.append(b1)
        board_history.append(b2)
        board_history.append(b3)
        board_history.append(b4)
        
        return (
            b1 in board_history, \
            # np.isin(b1, board_history).all(axis=1), \
            any(np.array_equal(b1, board) for board in board_history), 
            b5 in board_history, \
            # np.isin(b5, board_history).all(axis=1), \
            any(np.array_equal(b5, board) for board in board_history), 
        )

    
tb = TestBoard()
print(tb.isin_test())

# unittest.main()
