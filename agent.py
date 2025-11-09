import os
import uuid
import random
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque

from case_closed_game import Game, Direction, GameResult

# Flask API server setup
app = Flask(__name__)

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}

game_lock = Lock()
 
PARTICIPANT = "Sam Cakici"
AGENT_NAME = "John Tron"


def evaluate_danger(board, my_pos, opp_pos, open_space_count):
    """Evaluate danger level based on opponent proximity and available space.
    
    Returns danger level: 0 (safe), 1 (caution), 2 (danger), 3 (critical)
    """
    my_row, my_col = my_pos
    opp_row, opp_col = opp_pos
    
    # Calculate Manhattan distance to opponent
    distance_to_opponent = abs(my_row - opp_row) + abs(my_col - opp_col)
    
    # Danger factors:
    # 1. Opponent very close (within 3 cells)
    if distance_to_opponent <= 3:
        danger_level = 3  # Critical
    elif distance_to_opponent <= 5:
        danger_level = 2  # Danger
    else:
        danger_level = 1  # Caution
    
    # 2. Very limited space (less than 10 open cells nearby)
    if open_space_count < 10:
        danger_level = max(danger_level, 3)  # Critical
    elif open_space_count < 20:
        danger_level = max(danger_level, 2)  # Danger
    
    # 3. Check if opponent is approaching (could be enhanced with velocity)
    # For now, close proximity is the main factor
    
    return danger_level


def choose_best_move(state, my_agent, opponent_agent, boosts_remaining):
    """Choose the best move avoiding walls and maximizing open space away from opponent.
    
    Returns a move string like "UP" or "UP:BOOST".
    """
    if not state.get("board") or not my_agent.trail:
        return "UP"  # fallback
    
    board = state["board"]
    height = len(board)
    width = len(board[0]) if height > 0 else 0
    
    # Current position (head of trail)
    head = my_agent.trail[-1] if my_agent.trail else (0, 0)
    head_row, head_col = head
    
    # Opponent position
    opp_head = opponent_agent.trail[-1] if opponent_agent.trail else (height // 2, width // 2)
    opp_row, opp_col = opp_head
    
    # Define directions
    directions = {
        "UP": (-1, 0),
        "DOWN": (1, 0),
        "LEFT": (0, -1),
        "RIGHT": (0, 1)
    }
    
    valid_moves = []
    
    for direction, (dr, dc) in directions.items():
        new_row = head_row + dr
        new_col = head_col + dc
        
        # Check bounds
        if new_row < 0 or new_row >= height or new_col < 0 or new_col >= width:
            continue
        
        # Check if cell is free (only accept '.' or empty string)
        cell = board[new_row][new_col]
        if cell not in ('.', ''):
            continue
        
        # Score this direction
        # 1. Distance from opponent (higher is better)
        dist_to_opp = abs(new_row - opp_row) + abs(new_col - opp_col)
        
        # 2. Count open space in this direction (flood fill up to depth 5)
        open_space = count_open_space(board, new_row, new_col, depth=5)
        
        # Combined score: prioritize open space, then distance from opponent
        score = open_space * 10 + dist_to_opp
        
        valid_moves.append((direction, score))
    
    # If no valid moves, return a random direction (shouldn't happen often)
    if not valid_moves:
        return "UP"
    
    # Sort by score descending and pick best
    valid_moves.sort(key=lambda x: x[1], reverse=True)
    best_direction = valid_moves[0][0]
    best_open_space = valid_moves[0][1] // 10  # Extract open space count from score
    
    # Evaluate danger level
    danger_level = evaluate_danger(board, head, opp_head, best_open_space)
    
    # Add randomness when safe (danger level 0-1 and multiple good options)
    # Pick from top moves that have similar scores (within 20% of best)
    if danger_level <= 1 and len(valid_moves) > 1:
        best_score = valid_moves[0][1]
        threshold = best_score * 0.8  # 80% of best score
        good_moves = [move for move in valid_moves if move[1] >= threshold]
        
        # 30% chance to pick a random good move instead of the best
        if len(good_moves) > 1 and random.random() < 0.3:
            best_direction = random.choice(good_moves)[0]
    
    # Use boost in dangerous situations (levels 2-3) if available
    # Also use boost strategically in late game (turn > 30) when safe
    turn_count = state.get("turn_count", 0)
    use_boost = False
    
    if boosts_remaining > 0:
        # Critical/Danger: use boost to escape
        if danger_level >= 2:
            use_boost = True
        # Strategic boost in late game when safe
        elif turn_count > 30 and danger_level == 0:
            use_boost = True
    
    if use_boost:
        return f"{best_direction}:BOOST"
    
    return best_direction


def count_open_space(board, start_row, start_col, depth=5):
    """Count reachable open cells from (start_row, start_col) using BFS up to depth."""
    if start_row < 0 or start_row >= len(board) or start_col < 0 or start_col >= len(board[0]):
        return 0
    
    if board[start_row][start_col] not in ('.', ''):
        return 0
    
    visited = set()
    queue = deque([(start_row, start_col, 0)])
    visited.add((start_row, start_col))
    count = 0
    
    while queue:
        row, col, dist = queue.popleft()
        count += 1
        
        if dist >= depth:
            continue
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dr, col + dc
            
            if (new_row, new_col) in visited:
                continue
            
            if new_row < 0 or new_row >= len(board) or new_col < 0 or new_col >= len(board[0]):
                continue
            
            if board[new_row][new_col] not in ('.', ''):
                continue
            
            visited.add((new_row, new_col))
            queue.append((new_row, new_col, dist + 1))
    
    return count


@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint used by the judge to check connectivity.

    Returns participant and agent_name (so Judge.check_latency can create Agent objects).
    """
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


def _update_local_game_from_post(data: dict):
    """Update the local GLOBAL_GAME using the JSON posted by the judge.

    The judge posts a dictionary with keys matching the Judge.send_state payload
    (board, agent1_trail, agent2_trail, agent1_length, agent2_length, agent1_alive,
    agent2_alive, agent1_boosts, agent2_boosts, turn_count).
    """
    with game_lock:
        LAST_POSTED_STATE.clear()
        LAST_POSTED_STATE.update(data)

        if "board" in data:
            try:
                GLOBAL_GAME.board.grid = data["board"]
            except Exception:
                pass

        if "agent1_trail" in data:
            GLOBAL_GAME.agent1.trail = deque(tuple(p) for p in data["agent1_trail"]) 
        if "agent2_trail" in data:
            GLOBAL_GAME.agent2.trail = deque(tuple(p) for p in data["agent2_trail"]) 
        if "agent1_length" in data:
            GLOBAL_GAME.agent1.length = int(data["agent1_length"])
        if "agent2_length" in data:
            GLOBAL_GAME.agent2.length = int(data["agent2_length"])
        if "agent1_alive" in data:
            GLOBAL_GAME.agent1.alive = bool(data["agent1_alive"])
        if "agent2_alive" in data:
            GLOBAL_GAME.agent2.alive = bool(data["agent2_alive"])
        if "agent1_boosts" in data:
            GLOBAL_GAME.agent1.boosts_remaining = int(data["agent1_boosts"])
        if "agent2_boosts" in data:
            GLOBAL_GAME.agent2.boosts_remaining = int(data["agent2_boosts"])
        if "turn_count" in data:
            GLOBAL_GAME.turns = int(data["turn_count"])


@app.route("/send-state", methods=["POST"])
def receive_state():
    """Judge calls this to push the current game state to the agent server.

    The agent should update its local representation and return 200.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    _update_local_game_from_post(data)
    return jsonify({"status": "state received"}), 200


@app.route("/send-move", methods=["GET"])
def send_move():
    """Judge calls this (GET) to request the agent's move for the current tick.

    Query params the judge sends (optional): player_number, attempt_number,
    random_moves_left, turn_count. Agents can use this to decide.
    
    Return format: {"move": "DIRECTION"} or {"move": "DIRECTION:BOOST"}
    where DIRECTION is UP, DOWN, LEFT, or RIGHT
    and :BOOST is optional to use a speed boost (move twice)
    """
    player_number = request.args.get("player_number", default=1, type=int)

    with game_lock:
        state = dict(LAST_POSTED_STATE)   
        my_agent = GLOBAL_GAME.agent1 if player_number == 1 else GLOBAL_GAME.agent2
        opponent_agent = GLOBAL_GAME.agent2 if player_number == 1 else GLOBAL_GAME.agent1
        boosts_remaining = my_agent.boosts_remaining
   
    # -----------------your code here-------------------
    move = choose_best_move(state, my_agent, opponent_agent, boosts_remaining)
    # -----------------end code here--------------------

    return jsonify({"move": move}), 200


@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished and provides final state.

    We update local state for record-keeping and return OK.
    """
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)
    return jsonify({"status": "acknowledged"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5008"))
    app.run(host="0.0.0.0", port=port, debug=True)
