import cv2
import os
import numpy as np
import csv
from datetime import datetime

# --- Configuration ---
SCREENSHOT = "your_screenshot.png"
TEMPLATE_DIR = "letter_templates"
WORDS_FILE = "words.txt"

TILE_WIDTH = 189
TILE_HEIGHT = 189
GAP = 30
CROP_SIZE = 175
GRID_ROWS, GRID_COLS = 4, 4
x_start = 178
y_start = 1164

# --- Scoring system (based on standard Boggle rules) ---
def score_word(word):
    l = len(word)
    if l == 3:
        return 100
    elif l == 4:
        return 400
    elif l == 5:
        return 800
    elif l == 6:
        return 1400
    elif l == 7:
        return 1800
    elif l == 8:
        return 2200
    elif l == 9:
        return 2600
    elif l == 10:
        return 3000
    elif l == 11:
        return 3400
    elif l == 12:
        return 3800
    else:
        return 0  # ignore words shorter than 3 or longer than 12

# --- Load dictionary ---
with open(WORDS_FILE, "r") as f:
    DICTIONARY = set(word.strip().lower() for word in f)

# --- Load templates ---
templates = {}
for filename in os.listdir(TEMPLATE_DIR):
    if filename.endswith(".png"):
        letter = filename.split(".")[0].upper()
        template = cv2.imread(os.path.join(TEMPLATE_DIR, filename), cv2.IMREAD_GRAYSCALE)
        templates[letter] = template

# --- Load image and preprocess ---
image = cv2.imread(SCREENSHOT)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# --- Detect board ---
board = []
for row in range(GRID_ROWS):
    row_letters = []
    for col in range(GRID_COLS):
        tile_x = x_start + col * (TILE_WIDTH + GAP)
        tile_y = y_start + row * (TILE_HEIGHT + GAP)
        full_tile = gray[tile_y:tile_y + TILE_HEIGHT, tile_x:tile_x + TILE_WIDTH]
        offset_x = (TILE_WIDTH - CROP_SIZE) // 2
        offset_y = (TILE_HEIGHT - CROP_SIZE) // 2
        center_tile = full_tile[offset_y:offset_y + CROP_SIZE, offset_x:offset_x + CROP_SIZE]

        best_match = None
        best_score = -np.inf
        for letter, template in templates.items():
            res = cv2.matchTemplate(center_tile, template, cv2.TM_CCOEFF_NORMED)
            score = cv2.minMaxLoc(res)[1]
            if score > best_score:
                best_score = score
                best_match = letter

        row_letters.append(best_match)
    board.append(row_letters)

print("\n🧩 Detected Board:")
for row in board:
    print(" ".join(row))

# --- Boggle solver ---
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), 
              (0, -1),          (0, 1), 
              (1, -1),  (1, 0),  (1, 1)]

def in_bounds(x, y):
    return 0 <= x < 4 and 0 <= y < 4

def dfs(x, y, visited, current, found):
    current += board[y][x]
    visited.add((x, y))

    if current.lower() in DICTIONARY and len(current) >= 3:
        found.add(current.lower())

    for dx, dy in DIRECTIONS:
        nx, ny = x + dx, y + dy
        if in_bounds(nx, ny) and (nx, ny) not in visited:
            dfs(nx, ny, visited.copy(), current, found)

# --- Find all words ---
found_words = set()
for y in range(4):
    for x in range(4):
        dfs(x, y, set(), "", found_words)

# --- Calculate total possible score ---
max_score = sum(score_word(word) for word in found_words)

print(f"\n🧠 Max possible score: {max_score}")
print(f"📜 Found {len(found_words)} valid words")
print("🔡 Sample words:", sorted(list(found_words))[:10])

# --- Prompt for actual score ---
your_score = int(input("\nEnter your actual score from the game: "))
percent = round(your_score / max_score * 100, 2) if max_score > 0 else 0

print(f"\n📊 You scored {your_score} out of {max_score}")
print(f"🎯 That's {percent}% of the max score")

save_path = "games.csv"
with open(save_path, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        datetime.now().isoformat(),
        "".join("".join(row) for row in board),  # flatten 4x4 into 16-letter string
        ",".join(sorted(found)),
        score,
        max_score,
        percent
    ])
