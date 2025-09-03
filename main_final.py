import pygame
import numpy as np
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from maze import Maze
from agent_final import DQNAgent

# --- Constants ---
MAZE_WIDTH, MAZE_HEIGHT = 21, 21
CELL_SIZE = 20
MAX_STEPS_PER_EPISODE = 4000
VISION_SIZE = 5
FLAG_VALUE = 0.75

MAZE_WINDOW_SIZE = (MAZE_WIDTH * CELL_SIZE, MAZE_HEIGHT * CELL_SIZE)
NN_WINDOW_SIZE = (400, 300)
TOTAL_WIDTH = MAZE_WINDOW_SIZE[0] + NN_WINDOW_SIZE[0] + 30
TOTAL_HEIGHT = max(MAZE_WINDOW_SIZE[1], NN_WINDOW_SIZE[1])
GRAPH_X_OFFSET = MAZE_WINDOW_SIZE[0] + 15
BLACK, WHITE, GREEN, RED, BLUE, GRAY, YELLOW = (0,0,0), (255,255,255), (0,255,0), (255,0,0), (0,0,255), (100,100,100), (255,255,0)

pygame.init()
screen = pygame.display.set_mode((TOTAL_WIDTH, TOTAL_HEIGHT))
pygame.display.set_caption("Maze Solver (CNN with Randomized Goal)")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

def is_dead_end(grid, pos):
    (x, y) = pos
    if grid[y, x] != 0: return False
    neighbor_obstacles = 0
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nx, ny = x + dx, y + dy
        if not (0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0]) or grid[ny, nx] != 0:
            neighbor_obstacles += 1
    return neighbor_obstacles >= 3

# --- Sensory Functions ---
def get_local_vision(grid, pos):
    vision = np.ones((VISION_SIZE, VISION_SIZE), dtype=np.float32)
    center = VISION_SIZE // 2
    for r in range(VISION_SIZE):
        for c in range(VISION_SIZE):
            maze_r, maze_c = pos[1] + r - center, pos[0] + c - center
            if 0 <= maze_r < grid.shape[0] and 0 <= maze_c < grid.shape[1]:
                vision[r, c] = grid[maze_r, maze_c]
    vision[center, center] = 0.5 
    return vision.reshape(1, VISION_SIZE, VISION_SIZE, 1)

def cast_rays(grid, pos):
    (x, y) = pos
    distances = [0, 0, 0, 0]
    for i in range(1, y + 1):
        if grid[y - i, x] != 0: break
        distances[0] += 1
    for i in range(1, grid.shape[0] - y):
        if grid[y + i, x] != 0: break
        distances[1] += 1
    for i in range(1, grid.shape[1] - x):
        if grid[y, x + i] != 0: break
        distances[2] += 1
    for i in range(1, x + 1):
        if grid[y, x - i] != 0: break
        distances[3] += 1
    max_dist = max(MAZE_WIDTH, MAZE_HEIGHT)
    return (np.array(distances) / max_dist).reshape(1, 4)

def get_direction_to_goal(pos, goal_pos, maze_dims):
    dx = goal_pos[0] - pos[0]
    dy = goal_pos[1] - pos[1]
    norm_dx = dx / maze_dims[0]
    norm_dy = dy / maze_dims[1]
    return np.array([norm_dx, norm_dy]).reshape(1, 2)

def get_state(grid, pos, goal_pos, maze_dims):
    return [
        get_local_vision(grid, pos), 
        cast_rays(grid, pos), 
        get_direction_to_goal(pos, goal_pos, maze_dims)
    ]

# ... (All drawing functions are the same as the previous version) ...
def draw_maze(surface, maze_obj, episode_grid):
    for y, row in enumerate(episode_grid):
        for x, cell in enumerate(row):
            if cell == 1: pygame.draw.rect(surface, WHITE, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            elif cell == FLAG_VALUE: pygame.draw.rect(surface, YELLOW, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    start_rect = pygame.Rect(maze_obj.start_pos[0] * CELL_SIZE, maze_obj.start_pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(surface, RED, start_rect)
    end_rect = pygame.Rect(maze_obj.end_pos[0] * CELL_SIZE, maze_obj.end_pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(surface, GREEN, end_rect)
def draw_agent(surface, pos):
    agent_rect = pygame.Rect(pos[0] * CELL_SIZE + 2, pos[1] * CELL_SIZE + 2, CELL_SIZE - 4, CELL_SIZE - 4)
    pygame.draw.rect(surface, BLUE, agent_rect)
def draw_nn_graph(surface, scores, episode, total_episodes, current_score, epsilon):
    graph_rect = pygame.Rect(GRAPH_X_OFFSET, 0, NN_WINDOW_SIZE[0], NN_WINDOW_SIZE[1])
    pygame.draw.line(surface, GRAY, (graph_rect.left + 40, graph_rect.top + 10), (graph_rect.left + 40, graph_rect.bottom - 40), 1)
    pygame.draw.line(surface, GRAY, (graph_rect.left + 40, graph_rect.bottom - 40), (graph_rect.right - 10, graph_rect.bottom - 40), 1)
    title_text = font.render('Reward per Episode', True, WHITE)
    surface.blit(title_text, (graph_rect.centerx - title_text.get_width() // 2, graph_rect.top + 10))
    if len(scores) >= 2:
        max_score, min_score = max(1, max(scores)), min(0, min(scores))
        score_range = max_score - min_score if max_score > min_score else 1
        points = []
        plot_area_width, plot_area_height = graph_rect.width - 50, graph_rect.height - 50
        for i, score in enumerate(scores):
            x = graph_rect.left + 40 + (i / (len(scores) - 1)) * plot_area_width
            y = (graph_rect.bottom - 40) - ((score - min_score) / score_range) * (plot_area_height - 20)
            points.append((x, y))
        if len(points) > 1: pygame.draw.lines(surface, GREEN, False, points, 2)
    stats_y_start = graph_rect.bottom + 20
    episode_text = font.render(f"Episode: {episode + 1} / {total_episodes}", True, WHITE)
    surface.blit(episode_text, (graph_rect.left + 15, stats_y_start))
    score_text = font.render(f"Current Score: {current_score:.2f}", True, WHITE)
    surface.blit(score_text, (graph_rect.left + 15, stats_y_start + 25))
    epsilon_text = font.render(f"Epsilon: {epsilon:.2f}", True, WHITE)
    surface.blit(epsilon_text, (graph_rect.left + 15, stats_y_start + 50))


def main():
    try:
        maze = Maze(MAZE_WIDTH, MAZE_HEIGHT)
        agent = DQNAgent(
            vision_shape=(VISION_SIZE, VISION_SIZE, 1), 
            ray_shape=(4,), 
            direction_shape=(2,), 
            action_size=4
        )
        
        episodes = 1000
        batch_size = 64
        scores = []
        running = True
        for e in range(episodes):
            if not running: break
            maze.generate()
            episode_grid = maze.grid.astype(np.float32)
            agent_pos = list(maze.start_pos)
            
            state = get_state(episode_grid, agent_pos, maze.end_pos, (MAZE_WIDTH, MAZE_HEIGHT))
            
            total_reward = 0
            for time in range(MAX_STEPS_PER_EPISODE):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        running = False
                if not running: break

                action = agent.act(state)
                
                next_pos = agent_pos[:]
                if action == 0: next_pos[1] -= 1
                elif action == 1: next_pos[1] += 1
                elif action == 2: next_pos[0] -= 1
                elif action == 3: next_pos[0] += 1
                
                done = False
                hit_obstacle = not (0 <= next_pos[0] < MAZE_WIDTH and 0 <= next_pos[1] < MAZE_HEIGHT) or \
                               episode_grid[next_pos[1], next_pos[0]] != 0
                if tuple(next_pos) == maze.end_pos:
                    reward = 100
                    done = True
                elif tuple(next_pos) == maze.start_pos and time > 0:
                    reward = -15
                elif hit_obstacle:
                    reward = -10
                    is_near_start = abs(agent_pos[0] - maze.start_pos[0]) + abs(agent_pos[1] - maze.start_pos[1]) <= 1
                    if is_dead_end(episode_grid, agent_pos) and not is_near_start:
                        episode_grid[agent_pos[1], agent_pos[0]] = FLAG_VALUE
                    next_pos = agent_pos[:]
                else:
                    reward = -0.1
                
                total_reward += reward
                agent_pos = next_pos
                
                next_state = get_state(episode_grid, agent_pos, maze.end_pos, (MAZE_WIDTH, MAZE_HEIGHT))
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                
                screen.fill(BLACK)
                draw_maze(screen, maze, episode_grid)
                draw_agent(screen, agent_pos)
                draw_nn_graph(screen, scores, e, episodes, total_reward, agent.epsilon)
                pygame.display.update()
                clock.tick(120)

                if done: break
            
            if not running: break
            scores.append(total_reward)
            print(f"Episode: {e+1}/{episodes}, Score: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
            
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
    finally:
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    main()
