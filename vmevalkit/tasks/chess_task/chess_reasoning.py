#!/usr/bin/env python3
"""
Chess Reasoning Task for VMEvalKit

Self-contained chess mate-in-1 task generation system.
Follows the same data format as maze tasks with first/final frames and prompts.

Author: VMEvalKit Team
"""

import sys
import os
import random
from typing import List, Dict, Any, Optional, Set
from datetime import datetime

# Add python-chess to path
chess_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'submodules', 'python-chess')
if chess_path not in sys.path:
    sys.path.append(chess_path)

import chess
import chess.svg
from PIL import Image, ImageDraw, ImageFont
from .PROMPTS import PROMPTS


class SelfContainedMateGenerator:
    """Self-contained mate-in-1 position generator - no external dependencies."""
    
    def __init__(self):
        self.generated_positions = []
        self.position_hashes = set()
    
    def generate_mate_positions(self, num_positions: int = 150) -> List[Dict[str, Any]]:
        """Generate mate-in-1 positions using built-in templates."""
        print(f"🎯 Generating {num_positions} mate-in-1 positions...")
        
        # Generate using different strategies - EXPANDED TO 100+
        self._generate_back_rank_mates()      # ~50 positions
        self._generate_queen_corner_mates()   # ~30 positions  
        self._generate_simple_piece_mates()   # ~20 positions
        self._generate_rook_endgame_mates()   # ~15 positions
        self._generate_knight_mates()         # ~10 positions
        self._generate_position_variants()    # Double the positions
        
        # Return requested number
        available = min(num_positions, len(self.generated_positions))
        selected = random.sample(self.generated_positions, available)
        
        print(f"✅ Generated {len(selected)} verified mate positions")
        return selected
    
    def _add_position_if_valid(self, fen: str, mate_moves: List[str], 
                              description: str, tags: List[str], difficulty: str = "easy") -> bool:
        """Add position only if it's verified to work."""
        try:
            if fen in self.position_hashes:
                return False
            
            board = chess.Board(fen)
            valid_mates = []
            
            # Test each proposed mate move
            for mate_san in mate_moves:
                try:
                    move = board.parse_san(mate_san)
                    if move in board.legal_moves:
                        test_board = board.copy()
                        test_board.push(move)
                        if test_board.is_checkmate():
                            valid_mates.append(mate_san)
                except:
                    continue
            
            if not valid_mates:
                return False
            
            # Create puzzle data
            side_to_move = "white" if board.turn else "black"
            puzzle_id = f"chess_{len(self.generated_positions):04d}"
            
            puzzle_data = {
                "puzzle_id": puzzle_id,
                "fen": fen,
                "side_to_move": side_to_move,
                "mate_moves": valid_mates,
                "difficulty": difficulty,
                "description": description,
                "tags": tags
            }
            
            self.generated_positions.append(puzzle_data)
            self.position_hashes.add(fen)
            return True
            
        except Exception:
            return False
    
    def _generate_back_rank_mates(self):
        """Generate back-rank mate variations - EXPANDED TO 50+ POSITIONS."""
        print("🔥 Generating back-rank mates...")
        
        # MASSIVE expansion: All combinations of king positions, pawn structures, and pieces
        king_positions = [
            "k7", "1k6", "2k5", "3k4", "4k3", "5k2", "6k1", "7k"
        ]
        
        pawn_structures = [
            "ppp5", "1ppp4", "2ppp3", "3ppp2", "4ppp1", "5ppp",
            "pppp4", "1pppp3", "2pppp2", "3pppp1", "4pppp",
            "pp6", "1pp5", "2pp4", "3pp3", "4pp2", "5pp1", "6pp",
            "p7", "1p6", "2p5", "3p4", "4p3", "5p2", "6p1", "7p"
        ]
        
        attacking_pieces = [
            ("R6K", "Ra8#", "Rook"), ("Q6K", "Qa8#", "Queen"),
            ("1R5K", "Rb8#", "Rook"), ("1Q5K", "Qb8#", "Queen"),
            ("2R4K", "Rc8#", "Rook"), ("2Q4K", "Qc8#", "Queen")
        ]
        
        count = 0
        for king_pos in king_positions:
            for pawn_struct in pawn_structures:
                for piece_pos, move, piece_name in attacking_pieces:
                    fen = f"{king_pos}/{pawn_struct}/8/8/8/8/8/{piece_pos} w - - 0 1"
                    desc = f"{piece_name} back-rank mate vs {king_pos} with {pawn_struct}"
                    
                    if self._add_position_if_valid(fen, [move], desc, ["back_rank", piece_name.lower()]):
                        count += 1
        
        print(f"   Generated {count} back-rank positions")
    
    def _generate_queen_corner_mates(self):
        """Generate Queen+King corner mates - EXPANDED TO 30+ POSITIONS."""
        print("👑 Generating queen corner mates...")
        
        # Queen positions around corners
        queen_positions = [
            "6Q1", "5Q2", "4Q3", "3Q4", "2Q5", "1Q6", "Q7",
            "7Q", "6Q1", "5Q2"
        ]
        
        # Enemy king corner positions  
        enemy_king_positions = [
            "6k1", "7k", "5k2", "4k3"
        ]
        
        # White king support positions
        king_support_positions = [
            "6K1", "5K2", "7K", "4K3", "3K4"
        ]
        
        # Generate all combinations
        count = 0
        for enemy_king in enemy_king_positions:
            for queen_pos in queen_positions:
                for king_pos in king_support_positions:
                    fen = f"{enemy_king}/8/{queen_pos}/{king_pos}/8/8/8/8 w - - 0 1"
                    
                    # Try common queen mate moves
                    possible_moves = [
                        "Qa8#", "Qb8#", "Qc8#", "Qd8#", "Qe8#", "Qf8#", "Qg8#", "Qh8#",
                        "Qa7#", "Qb7#", "Qc7#", "Qd7#", "Qe7#", "Qf7#", "Qg7#", "Qh7#"
                    ]
                    
                    for move in possible_moves:
                        desc = f"Queen corner mate: {enemy_king} vs Q+K"
                        if self._add_position_if_valid(fen, [move], desc, ["queen", "corner"]):
                            count += 1
                            break  # Only need one working move per position
        
        print(f"   Generated {count} queen corner positions")
    
    def _generate_simple_piece_mates(self):
        """Generate simple piece mates - EXPANDED TO 20+ POSITIONS."""
        print("⚡ Generating simple mates...")
        
        # White simple mates
        white_patterns = [
            ("k7", "Q6K", "Qa8#", "Simple queen file mate"),
            ("1k6", "Q6K", "Qb8#", "Queen b-file mate"),
            ("2k5", "Q6K", "Qc8#", "Queen c-file mate"),
            ("k7", "R6K", "Ra8#", "Simple rook file mate"),
            ("1k6", "R6K", "Rb8#", "Rook b-file mate"),
            ("k7", "1Q5K", "Qb8#", "Queen mate shifted"),
            ("7k", "6QK", "Qg8#", "Queen g-file mate"),
        ]
        
        # Black simple mates (for variety)
        black_patterns = [
            ("6qK", "6k1", "Qg7#", "Black queen mate"),
            ("5q1K", "6k1", "Qf7#", "Black queen f-file mate"), 
            ("r6K", "6k1", "Ra7#", "Black rook mate"),
            ("1r5K", "6k1", "Rb7#", "Black rook b-file mate"),
            ("6qK", "5k2", "Qf7#", "Black queen vs f8 king"),
            ("q6K", "6k1", "Qa7#", "Black queen a-file mate"),
            ("4q2K", "6k1", "Qe7#", "Black queen e-file mate"),
        ]
        
        count = 0
        
        # Generate white mates
        for king_pos, piece_pos, move, desc in white_patterns:
            fen = f"{king_pos}/8/8/8/8/8/8/{piece_pos} w - - 0 1"
            piece_name = "queen" if "Q" in piece_pos else "rook"
            if self._add_position_if_valid(fen, [move], desc, [piece_name, "simple"]):
                count += 1
        
        # Generate black mates  
        for piece_line, king_line, move, desc in black_patterns:
            fen = f"{piece_line}/8/{king_line}/8/8/8/8/8 b - - 0 1"
            piece_name = "queen" if "q" in move.lower() else "rook"
            if self._add_position_if_valid(fen, [move], desc, [piece_name, "black_to_move"]):
                count += 1
        
        print(f"   Generated {count} simple piece positions")
    
    def _generate_rook_endgame_mates(self):
        """Generate Rook+King vs King endgame mates - 15+ POSITIONS."""
        print("🏰 Generating rook endgame mates...")
        
        # Rook + King coordination patterns
        rook_patterns = [
            ("k7", "R7", "K7", "Ra8#", "Rook file mate with K support"),
            ("1k6", "R7", "K7", "Rb8#", "Rook b-file mate"),
            ("k7", "1R6", "K7", "Rb8#", "Rook shifted position"),
            ("7k", "6R1", "6K1", "Rg8#", "Rook g-file mate"),
            ("6k1", "6R1", "6K1", "Rg8#", "Rook restricting king"),
            ("k7", "R7", "1K6", "Ra8#", "Rook with K on b7"),
            ("k7", "R7", "2K5", "Ra8#", "Rook with K on c7"),
            ("2k5", "R7", "K7", "Rc8#", "Rook c-file coordination"),
            ("3k4", "R7", "K7", "Rd8#", "Rook d-file coordination"),
        ]
        
        count = 0
        for king_pos, rook_pos, white_king_pos, move, desc in rook_patterns:
            fen = f"{king_pos}/{rook_pos}/{white_king_pos}/8/8/8/8/8 w - - 0 1"
            if self._add_position_if_valid(fen, [move], desc, ["rook", "endgame"]):
                count += 1
        
        print(f"   Generated {count} rook endgame positions")
    
    def _generate_knight_mates(self):
        """Generate Knight mate patterns - 10+ POSITIONS."""
        print("🐴 Generating knight mates...")
        
        # Knight fork and smothered mate patterns
        knight_patterns = [
            ("r5k1", "6N1", "7K", "Ne7#", "Knight fork checkmate"),
            ("6rk", "5N1p", "6pK", "Ng5#", "Knight smothered attempt"), 
            ("4k3", "3N4", "7K", "Nf6#", "Knight fork center"),
            ("2k5", "3N4", "7K", "Nd6#", "Knight d6 mate"),
            ("7k", "5N2", "7K", "Nf7#", "Knight f7 mate"),
            ("k7", "1N6", "7K", "Nc8#", "Knight c8 mate"),
            ("1k6", "2N5", "7K", "Nd7#", "Knight d7 mate"),
        ]
        
        count = 0
        for enemy_line, knight_line, king_line, move, desc in knight_patterns:
            fen = f"{enemy_line}/8/{knight_line}/{king_line}/8/8/8/8 w - - 0 1"
            if self._add_position_if_valid(fen, [move], desc, ["knight", "tactical"]):
                count += 1
        
        print(f"   Generated {count} knight mate positions")
    
    def _generate_position_variants(self):
        """Generate variants through position transformations.""" 
        print("🔄 Generating position variants...")
        
        original_count = len(self.generated_positions)
        original_positions = self.generated_positions[:]
        
        for puzzle in original_positions:
            # Try horizontal mirroring
            try:
                mirrored_fen = self._mirror_fen_horizontal(puzzle["fen"])
                if mirrored_fen and mirrored_fen != puzzle["fen"]:
                    mirrored_moves = [self._mirror_move_horizontal(m) for m in puzzle["mate_moves"]]
                    desc = puzzle["description"] + " (mirrored)"
                    tags = puzzle["tags"] + ["mirrored"]
                    
                    self._add_position_if_valid(mirrored_fen, mirrored_moves, desc, tags, puzzle["difficulty"])
            except:
                continue
        
        variant_count = len(self.generated_positions) - original_count
        print(f"   Generated {variant_count} position variants")
    
    def _mirror_fen_horizontal(self, fen: str) -> Optional[str]:
        """Mirror FEN horizontally (a↔h files)."""
        try:
            parts = fen.split()
            board = parts[0]
            ranks = board.split('/')
            mirrored_ranks = [rank[::-1] for rank in ranks]
            mirrored_board = '/'.join(mirrored_ranks)
            return mirrored_board + " " + " ".join(parts[1:])
        except:
            return None
    
    def _mirror_move_horizontal(self, move: str) -> str:
        """Mirror move horizontally."""
        file_map = {'a': 'h', 'b': 'g', 'c': 'f', 'd': 'e',
                   'e': 'd', 'f': 'c', 'g': 'b', 'h': 'a'}
        result = ""
        for char in move:
            result += file_map.get(char, char)
        return result


def generate_chess_board_png(fen: str, output_path: str, board_size: int = 400) -> bool:
    """Generate PNG representation of chess board (matching maze format)."""
    board = chess.Board(fen)
    svg_content = chess.svg.board(board=board, size=board_size)
    
    # Prefer high-fidelity SVG→PNG when available
    try:
        import cairosvg  # type: ignore
        cairosvg.svg2png(bytestring=svg_content.encode('utf-8'), write_to=output_path)
        return True
    except Exception:
        # Fallback: pure-PIL rasterizer (no native deps). Draw simple board with glyph pieces.
        _render_board_png_with_pil(board, output_path, board_size)
        return True


def _render_board_png_with_pil(board: chess.Board, output_path: str, board_size: int = 400) -> None:
    """Render a simple chessboard PNG using PIL (no Cairo dependency)."""
    img = Image.new("RGB", (board_size, board_size), color="white")
    draw = ImageDraw.Draw(img)
    square_px = board_size // 8
    light = (240, 217, 181)
    dark = (181, 136, 99)
    text_color_white = (245, 245, 245)
    text_color_black = (20, 20, 20)

    # Attempt to load a larger font; fallback to default
    try:
        font_size = int(square_px * 0.8)
        # DejaVuSans includes Unicode chess glyphs (U+2654..U+265F)
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        # Fallback to default font; glyph quality may vary
        font = ImageFont.load_default()

    # Draw squares
    for rank in range(8):
        for file in range(8):
            x0 = file * square_px
            y0 = (7 - rank) * square_px  # rank 0 at bottom
            color = light if (rank + file) % 2 == 0 else dark
            draw.rectangle([x0, y0, x0 + square_px, y0 + square_px], fill=color)

    # Unicode glyphs for chess pieces (white: ♔♕♖♗♘♙, black: ♚♛♜♝♞♟)
    unicode_map = {
        'P': '\u2659', 'N': '\u2658', 'B': '\u2657', 'R': '\u2656', 'Q': '\u2655', 'K': '\u2654',
        'p': '\u265F', 'n': '\u265E', 'b': '\u265D', 'r': '\u265C', 'q': '\u265B', 'k': '\u265A',
    }

    # Draw pieces using glyphs (fallback to letters if glyph unsupported)
    piece_map = board.piece_map()
    for sq, piece in piece_map.items():
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        x_center = file * square_px + square_px // 2
        y_center = (7 - rank) * square_px + square_px // 2
        sym = piece.symbol()
        label = unicode_map.get(sym, sym.upper())
        # Measure text size for centering
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
        except Exception:
            w, h = draw.textlength(label, font=font), int(square_px * 0.7)
        # Choose fill and outline for contrast
        fill_color = text_color_white if piece.color else text_color_black
        outline_color = (0, 0, 0) if piece.color else (255, 255, 255)
        x = x_center - w / 2
        y = y_center - h / 2
        # Simple outline by drawing around the target point
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text((x + dx, y + dy), label, font=font, fill=outline_color)
        draw.text((x, y), label, font=font, fill=fill_color)
    img.save(output_path, format="PNG")


def create_chess_task_pair(puzzle_data: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    """
    Create a chess task pair in the same format as maze tasks.
    
    Args:
        puzzle_data: Chess puzzle information
        task_id: Unique task identifier
        
    Returns:
        Task pair dictionary matching maze format
    """
    
    # Generate standardized prompt (using PROMPTS[0] as default)
    side = "White" if puzzle_data["side_to_move"] == "white" else "Black"
    prompt = PROMPTS[0].format(side=side)
    
    # Create temporary files that will be moved to per-question folders
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    first_temp_path = os.path.join(temp_dir, f"{task_id}_first.png")
    final_temp_path = os.path.join(temp_dir, f"{task_id}_final.png")
    
    # Generate first frame (initial position)
    generate_chess_board_png(puzzle_data["fen"], first_temp_path)
    
    # Generate final frame (after mate move)
    board = chess.Board(puzzle_data["fen"])
    mate_move = puzzle_data["mate_moves"][0]
    move = board.parse_san(mate_move)
    board.push(move)
    
    generate_chess_board_png(board.fen(), final_temp_path)
    
    # Paths will be updated when moved to per-question folders
    first_image_path = first_temp_path
    final_image_path = final_temp_path
    
    print(f"✅ Created chess task {task_id}: {puzzle_data['description']}")
    
    # Create task pair in EXACT same format as maze tasks
    task_pair = {
        "id": task_id,
        "prompt": prompt,
        "first_image_path": first_image_path,
        "final_image_path": final_image_path,
        "task_category": "Chess",
        "chess_data": {
            "generation_method": "mate_in_1_templates",
            "initial_fen": puzzle_data["fen"],
            "mate_moves": puzzle_data["mate_moves"],
            "pattern_tags": puzzle_data.get("tags", [])
        },
        "difficulty": puzzle_data.get("difficulty", "easy"),
        "side_to_move": puzzle_data["side_to_move"],
        "mate_moves": puzzle_data["mate_moves"],  # For compatibility
        "pattern_description": puzzle_data["description"],
        "created_at": datetime.now().isoformat()
    }
    
    return task_pair


def create_dataset(num_samples: int = 100) -> Dict[str, Any]:
    """
    Create chess reasoning dataset in EXACT same format as maze dataset.
    
    Args:
        num_samples: Number of chess tasks to generate
        
    Returns:
        Dataset dictionary matching maze format exactly
    """
    
    print(f"🎯 Creating chess dataset with {num_samples} samples...")
    print("=" * 50)
    
    # Generate mate positions using built-in generator
    generator = SelfContainedMateGenerator()
    positions = generator.generate_mate_positions(num_samples)
    
    if not positions:
        print("❌ No positions generated!")
        return {"name": "chess_tasks", "description": "Failed to generate", "pairs": []}
    
    # Create task pairs
    pairs = []
    for i, puzzle_data in enumerate(positions):
        task_id = f"chess_{i:04d}"
        try:
            task_pair = create_chess_task_pair(puzzle_data, task_id)
            pairs.append(task_pair)
        except Exception as e:
            print(f"❌ Failed to create task {task_id}: {e}")
    
    # Create dataset in EXACT same format as maze dataset
    dataset = {
        "name": "chess_tasks", 
        "description": f"Chess mate-in-1 reasoning tasks for video model evaluation ({len(pairs)} pairs)",
        "pairs": pairs
    }
    
    # Don't save to intermediate folder anymore - will be handled by create_dataset.py
    print(f"🎯 Chess dataset created successfully!")
    print(f"   Total tasks: {len(pairs)}")
    print(f"   Images created: {len(pairs) * 2}")
    
    return dataset


# Dataset creation should only be done via vmevalkit/runner/create_dataset.py
# This module only provides the create_dataset() function as an API