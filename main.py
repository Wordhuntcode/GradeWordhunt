from flask import Flask, request, render_template
import cv2
import os
import numpy as np
from werkzeug.utils import secure_filename
import csv
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Config ---
TILE_WIDTH = 189
TILE_HEIGHT = 189
GAP = 30
CROP_SIZE = 175
x_start = 178
y_start = 1164

# --- Load dictionary ---
with open("words.txt", "r") as f:
    DICTIONARY = set(word.strip().lower() for word in f)

# --- Load templates ---
TEMPLATE_DIR = "letter_templates"
templates = {}
for filename in os.listdir(TEMPLATE_DIR):
    if filename.endswith(".png"):
        letter = filename.split(".")[0].upper()
        path = os.path.join(TEMPLATE_DIR, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        templates[letter] = img

# --- Scoring ---
def score_word(word):
    l = len(word)
    return {
        3: 100, 4: 400, 5: 800, 6: 1400, 7: 1800,
        8: 2200, 9: 2600, 10: 3000, 11: 3400, 12: 3800
    }.get(l, 0)

DIRECTIONS = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]

def in_bounds(x, y): return 0 <= x < 4 and 0 <= y < 4

def dfs(x, y, visited, current, found, board):
    current += board[y][x]
    visited.add((x, y))
    if current.lower() in DICTIONARY and len(current) >= 3:
        found.add(current.lower())
    for dx, dy in DIRECTIONS:
        nx, ny = x + dx, y + dy
        if in_bounds(nx, ny) and (nx, ny) not in visited:
            dfs(nx, ny, visited.copy(), current, found, board)

def extract_board(image_path):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    board = []
    for row in range(4):
        row_letters = []
        for col in range(4):
            tx = x_start + col * (TILE_WIDTH + GAP)
            ty = y_start + row * (TILE_HEIGHT + GAP)
            full_tile = gray[ty:ty + TILE_HEIGHT, tx:tx + TILE_WIDTH]
            ox = (TILE_WIDTH - CROP_SIZE) // 2
            oy = (TILE_HEIGHT - CROP_SIZE) // 2
            center_tile = full_tile[oy:oy + CROP_SIZE, ox:ox + CROP_SIZE]

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
    return board

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image = request.files["screenshot"]
        score = int(request.form["actual_score"])

        filename = secure_filename(image.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        image.save(filepath)

        board = extract_board(filepath)

        # Find words
        found = set()
        for y in range(4):
            for x in range(4):
                dfs(x, y, set(), "", found, board)

        max_score = sum(score_word(w) for w in found)
        percent = round(score / max_score * 100, 2) if max_score else 0

        # Save to CSV
        with open("games.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                "".join("".join(row) for row in board),
                ",".join(sorted(found)),
                score,
                max_score,
                percent
            ])

        return render_template("index.html",
                               board=board,
                               max_score=max_score,
                               your_score=score,
                               percent=percent,
                               words=sorted(found))

    return render_template("index.html")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
