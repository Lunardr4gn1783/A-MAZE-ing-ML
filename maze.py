import numpy as np
import random

class Maze:
    """
    Generates a perfect maze using a recursive backtracking algorithm.
    The start and end points are randomized for each generation.
    """
    def __init__(self, width=21, height=21):
        self.width = width if width % 2 != 0 else width + 1
        self.height = height if height % 2 != 0 else height + 1
        self.grid = np.ones((self.height, self.width), dtype=np.uint8)
        self.start_pos = (0, 0) # Will be set during generation
        self.end_pos = (0, 0)   # Will be set during generation

    def generate(self):
        """Carves the maze and sets random start/end points."""
        # Carving process using recursive backtracking
        self.grid.fill(1)
        start_x, start_y = (random.randrange(1, self.width, 2), 
                            random.randrange(1, self.height, 2))
        self.grid[start_y, start_x] = 0
        stack = [(start_x, start_y)]
        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 < nx < self.width-1 and 0 < ny < self.height-1 and self.grid[ny, nx] == 1:
                    neighbors.append((nx, ny))
            if neighbors:
                nx, ny = random.choice(neighbors)
                self.grid[ny, nx] = 0
                self.grid[cy + (ny - cy) // 2, cx + (nx - cx) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()

        # --- New: Randomize start and end positions ---
        path_cells = np.argwhere(self.grid == 0)
        
        # Ensure start and end are sufficiently far apart
        min_distance = (self.width + self.height) / 2.5 # Heuristic for min distance
        while True:
            # Pick two unique random indices from the path cells
            start_idx, end_idx = random.sample(range(len(path_cells)), 2)
            start_y, start_x = path_cells[start_idx]
            end_y, end_x = path_cells[end_idx]
            
            # Calculate Manhattan distance (distance in grid steps)
            distance = abs(start_x - end_x) + abs(start_y - end_y)
            
            if distance >= min_distance:
                break
                
        self.start_pos = (int(start_x), int(start_y))
        self.end_pos = (int(end_x), int(end_y))
