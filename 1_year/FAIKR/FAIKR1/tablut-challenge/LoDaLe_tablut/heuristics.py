from board import *

INF = 1_000_000

def heuristic_1(board: Board, weights=None) -> int:
    blacks = board.get_all_pieces_of_color(BLACK)
    num_blacks = len(blacks)
    return -num_blacks  # plays for black

def heuristic_2(board: Board, weights=None) -> int:
    whites = board.get_all_pieces_of_color(WHITE)
    num_whites = len(whites)

    return num_whites   # plays for white

def heuristic_3(board: Board, weights=None) -> int:
    surrounding_escapes = 0
    surrounding_blacks = 0
    king_pos = board.get_king()
    for j in range(1,3):
        king_ring = board.ring_occupation(king_pos, j)["str"]
        for i in range(4):
            if king_ring[i*2] == ESCAPE[0]:
                surrounding_escapes += 1
            if king_ring[i*2] == BLACK[0]:
                surrounding_blacks += 1
    return surrounding_escapes - surrounding_blacks     # >0 for W and <0 for B
    

def grey_heuristic(board: Board, depth:int) -> int:
    ''' 
    MAX = white

    MIN = black
    '''

    ### ALL VARIABLES ##################################################################################################################
    king_pos = None                     # 1. Check if the player has won or lost
    free_routes_to_escapes = 0          # 2. How many free routes will have the king to reach the escapes
    free_routes = 0                     # 3. How many free routes will have the king
    surrounding_protection = 0          # 4. How well the king is protected by whites (with adjacent whites)
    surrounding_threats = 0             # 5. How much in danger (has non-white obstacles) is the king (with adjacent black or camps)
    possible_protection = 0             # 6. How well the king is protected by whites (untill the first obstalces)
    possible_checkers = 0               # 7. How many black can be adjacent to the king the next move
    possible_threats = 0                # 8. How much in danger (has non-white obstacles) is the king (untill the first obstalces)
    num_rombus_pos = 0                  # 9. The black must tend to have a rombus disposition
    num_covered_escapes = 0             # 10. The black may tend to cover the escapes
    num_double_covered_escapes = 0      # 11. The black should prefer the double cover escapes (covers 2 with 1 pawn)
    # freecell_near_escapes = 0         # 12. The white should tend to cover free cells near escapes before black
    # white_in_highlighted_cells = 0    # 12. Escapes which are covered first by a white
    num_whites = 0                      # 13. The number of whites
    num_blacks = 0                      # 14. The number of blacks
    ####################################################################################################################################


    ### CHECK THE TABLE ###################################################################################################
    whites = board.get_all_pieces_of_color(WHITE)
    blacks = board.get_all_pieces_of_color(BLACK)
    num_whites = len(whites)
    num_blacks = len(blacks)
    king_pos = board.get_king()
    whites = np.array(whites)
    blacks = np.array(blacks)


    intersection_rombus = np.isin(blacks, ROMBUS_POS).all(axis=1)
    num_rombus_pos = np.sum(intersection_rombus)
    intersection_covered_escapes = np.isin(blacks, ESCAPE_COVER).all(axis=1)
    num_covered_escapes = np.sum(intersection_covered_escapes)
    intersection_double_covered_escapes = np.isin(blacks, DOUBLE_ESCAPE_COVER).all(axis=1)
    num_double_covered_escapes = np.sum(intersection_double_covered_escapes)
    ########################################################################################################################

    
    # If the king got captured ==> white lost
    if board.is_king_escaped():
        return INF - depth
    
    if board.is_king_captured():
        return -INF + depth
    

    ### CHECK ALL AROUND THE KING ###########################################################################################
    king_ring = board.ring_occupation(king_pos, 1)["str"]
    for i in range(4):
        if king_ring[i*2] == WHITE[0]:
            surrounding_protection += 1
        if king_ring[i*2] == BLACK[0] or king_ring[i*2] == CAMP[0]:
            surrounding_threats += 1

    king_x, king_y = king_pos
    checking_start = [(king_x-1, king_y), (king_x, king_y+1), (king_x+1, king_y), (king_x, king_y-1)]
    checking_arrive = [(0, king_y),(king_x, 8), (8, king_y), (king_x, 0)]
    temp_arrive = [(king_x-1, 0), (0, king_y+1), (king_x+1, 8), (0, king_y-1), (king_x-1, 8), (8, king_y+1), (king_x+1, 0), (8, king_y-1)]
    # Check the column and the row of the king
    for i in range(4):
        column_check = board.segment_occupation(king_pos, checking_arrive[i])["str"]
        if  len(column_check)>0 and \
            column_check[-1] == ESCAPE[0] and \
            (all(c == EMPTY[0] for c in column_check[:-1]) or len(column_check) == 1):
                
            free_routes_to_escapes += 1
            
        else:
            for car in column_check:
                if car == BLACK[0]:
                    possible_checkers += 1
                    possible_threats += 1
                    break
                elif car == WHITE[0]:
                    possible_protection += 1
                    break
    # Check the column and the row near the king
    for i in range(8):
        column_check = board.segment_occupation(checking_start[i%2], temp_arrive[i])["str"]
        for car in column_check:
            if car == BLACK[0]:
                possible_checkers += 1
                break
            elif car == WHITE[0]:
                possible_protection += 1
                break
            elif car == ESCAPE[0] or car == CAMP[0]:
                break
    #############################################################################################################################



    ### Check if king is in highlighted cells ####################################################################################
    highlighted_cells = board.get_highlighted_escape_cells()
    king_in_highlighted_cells = any((king_pos == cell) for cell in highlighted_cells)
    ##############################################################################################################################


    free_routes = 4 - possible_threats - possible_protection - surrounding_threats - surrounding_protection


    ### Compute scorings (DECIDE THE WEIGHTS also using non linear functions) #####################################

    # Use weight to optimize the module of the score
    # W1 = 3    # Bilancia la distanza dall'escape
    W2 = 800   # Diminuisce l'enfasi sui percorsi liberi verso l'escape
    W3 = 1      # Incentiva percorsi liberi solo se l'escape non è immediato
    W4 = 10     # Migliora la protezione del re
    W5 = 20     # Diminuire l'effetto negativo delle minacce
    W6 = 5      # Diminuire l'effetto negativo delle minacce più lontane
    W7 = 1      # Migliora leggermente la protezione lontana
    W8 = 10     # Riduce l'enfasi su rombo
    W9 = 5      # Diminuire l'effetto negativo di escape coperti
    W10 = 5     # Riduce l'enfasi su doppi escape coperti
    W11 = 25    # Considera il numero di bianchi vivi
    W12 = 10    # Considera il numero di neri vivi
    W13 = 5     # Porta il re in buone posizioni vicino agli escapes (highlighted cells)
    # Use a dict for better usability and debug
    scorings = {

        # The king should consider the nearest escape only if it is not surrounded by blacks
        # in that direction (???)
        # 'nearest_escape_distance_score':
            # W1 * (math.sqrt(50) - get_min_escape_dist())    if free_routes_to_escapes==0
                                                            # else 0,

        # The king must always consider options where he can win in one move
        # NOTE: forse potremmo usare direttamente +infinito ma così magari eviteremmo
        # scelte in cui ci sono più di 2 vie libere (btw vittoria assicurata)
        'free_routes_to_escapes_score': INF-depth-1 if free_routes_to_escapes>0 else 0,

        # The king must consider free routes that don't lead to victory only if
        # there are not free routes to escapes
        # NOTE: dobbiamo mettere che altrimenti il peso è negativo perchè altrimenti lo score
        # resterebbe più o meno uguale e dobbiamo rimarcare che sia una brutta idea
        'free_routes_score':
            W3 * free_routes    if free_routes_to_escapes==0
                                else - W3 * free_routes,

        # The king should consider how well he is protected only if he is not in a winning
        # position, so if he is not that much surrounded by blacks and he is next to an escape
        # NOTE: dobbiamo mettere che altrimenti il peso è negativo perchè altrimenti lo score
        # resterebbe più o meno uguale e dobbiamo rimarcare che sia una brutta idea
        'surrounding_protection_score':
            W4 * surrounding_protection     if free_routes_to_escapes == 0
                                            else - 0.5 * W4 * surrounding_protection,

        # The king should consider how much he is in danger only if he is not in a winning
        # position, so if he is not that much surrounded by blacks and he is next to an escape
        'surrounding_threats_score':
            - W5 * surrounding_threats  if free_routes_to_escapes == 0
                                        else 0,

        # The king should consider if in the next move a blacks can became adjacent
        # NOTE: ancora meno importante se il re è vicino agli escapes
        'possible_threats_score':
            - W6 * possible_checkers     if free_routes_to_escapes == 0
                                        else 0,

        # The king should consider if in the next move a white can became adjacent
        # NOTE: ancora meno importante se il re è vicino agli escapes
        'possible_protection_score':
            W7 * possible_protection    if free_routes_to_escapes == 0
                                        else - W7 * possible_protection,

        # The king must consider them only if he is inside the rombus and if the blacks are enough to create the rombus
        'rombus_position_score':
            - W8 * num_rombus_pos       if free_routes_to_escapes == 0 and num_blacks > 8
                                        else 0,

        # The king must consider them only if he has no other escapes
        # NOTE: servono pesi alti
        'covered_escapes_score':
            - W9 * num_covered_escapes  if free_routes_to_escapes == 0
                                        else 0,

        # The king must consider them only if he has no other escapes
        # NOTE: servono pesi molto molto alti
        'double_covered_escapes_score':
            - W10 * num_double_covered_escapes  if free_routes_to_escapes == 0
                                                else 0,

        # The player must consider the number of white alive only if he is not in
        # a winning position
        'white_alive' :
            W11 * num_whites   if free_routes_to_escapes == 0
                                            else 0,
        
        # The player must consider the number of black alive only if he is not in
        # a winning position
        'black_alive' :
            W12 * num_blacks if free_routes_to_escapes == 0
                                            else 0,
        # The player must consider the escape protection only if he is not in
        # a winning position
        'escape_protection' :
            W13                             if free_routes_to_escapes == 0 and king_in_highlighted_cells
                                                else 0
    }

    return sum(scorings.values())