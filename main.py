import cv2
import numpy as np
import pyautogui
import time
import os

# -------------------------------
# Setup template path safely
# -------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))  # folder of this script
template_path = os.path.join(script_dir, 'skipbtn.png')

# Load the skip button template
template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
if template is None:
    raise FileNotFoundError(f"Template not found at: {template_path}")

# Convert template to grayscale (handles alpha channel too)
if template.shape[2] == 4:  # has alpha
    template = cv2.cvtColor(template, cv2.COLOR_BGRA2GRAY)
else:
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

w, h = template.shape[1], template.shape[0]

print("Template loaded successfully. Starting ad skipper...")

# -------------------------------
# Main loop
# -------------------------------
while True:
    # Take a screenshot
    screenshot = pyautogui.screenshot()
    
    # Convert screenshot to grayscale
    screenshot_gray = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)
    
    # Match the template
    result = cv2.matchTemplate(screenshot_gray, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    
    # If match is strong enough
    if max_val > 0.8:
        print(f"Ad detected! Clicking skip at {max_loc} (confidence: {max_val:.2f})")
        pyautogui.click(max_loc[0] + w//2, max_loc[1] + h//2)
        time.sleep(2)  # wait to avoid multiple clicks
    
    time.sleep(0.5)  # check twice per second
