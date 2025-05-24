import socket
import json
import struct
import numpy as np

class SocketManager:
    
    def __init__(self, ip, port, player_name):
        self.name = player_name
        self._ip = ip
        self._port = port
        self._sock = None

    def create_socket(self):
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.setblocking(1)
        except socket.error as e:
            print(f"Error while creating the socket: {e}")
            self._sock = None

    def connect(self):
        if self._sock is None:
            print("Socket not initialized yet.")
            return
        try:
            self._sock.connect((self._ip, self._port))
            print(f"Connected to the server {self._ip}:{self._port}")
            
            # Send the player's name to the server
            self._sock.send(struct.pack('>i', len(self.name)))
            self._sock.send(self.name.encode())
            
        except socket.error as e:
            print(f"Error while connecting: {e}")
            self._sock.close()
            self._sock = None
    
    def close_socket(self):
        if self._sock:
            self._sock.close()
            print("Socket closed.")
            
    def recvall(self, n):
        # Helper function to recv n bytes or return None if EOF is hit
        data = b''
        while len(data) < n:
            packet = self._sock.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data
    
    def get_state(self):
        len_bytes = struct.unpack('>i', self.recvall(4))[0]
        current_state_server_bytes = self._sock.recv(len_bytes)
        # print("Ricevuti ", len(current_state_server_bytes))

        # Converting byte into json
        json_current_state_server = json.loads(current_state_server_bytes) 
                
        # Convert the board matrix in numpy matrix 
        state = dict(json_current_state_server).copy()
        state['board'] = np.array(json_current_state_server['board'])   
        
        return state
    
    def send_move(self, move):
        _from, _to, turn = move

        move = json.dumps({"from": _from, "to": _to, "turn": turn})

        self._sock.send(struct.pack('>i', len(move)))
        self._sock.send(move.encode())
        return move
        