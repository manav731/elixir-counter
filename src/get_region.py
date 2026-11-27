import pyautogui
import time

print("Step 1: Move your mouse to the TOP-LEFT corner of the game window.")
time.sleep(4)  

x1, y1 = pyautogui.position()
print(f"Top-left recorded at: ({x1}, {y1})")

print("Step 2: Move your mouse to the BOTTOM-RIGHT corner of the game window.")
time.sleep(4)  

x2, y2 = pyautogui.position()
print(f"Bottom-right recorded at: ({x2}, {y2})")

width = x2 - x1
height = y2 - y1

print("\nRun the tracker with this region:")
print(f"python src/motion_detect_demo.py --region {y1},{x1},{width},{height}")
