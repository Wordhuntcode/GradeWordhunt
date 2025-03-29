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

DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),          (0, 1),
              (1, -1),  (1, 0),  (1, 1)]
