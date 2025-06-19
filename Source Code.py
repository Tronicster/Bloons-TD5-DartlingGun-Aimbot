import cv2
import numpy as np
import mss
import pyautogui
import threading
import time
import tkinter as tk
from tkinter import ttk, scrolledtext, BooleanVar
from PIL import Image, ImageTk
import random

# HSV color ranges for bloons
COLOR_RANGES = {
    "Blue":   ([90, 80, 100], [140, 255, 255]),
    "Red":    ([0, 120, 100], [10, 255, 255]),
    "Green":  ([50, 150, 50], [70, 255, 255]),
    "Yellow": ([22, 180, 180], [32, 255, 255]),
    "Pink":   ([145, 100, 100], [170, 255, 255]),
    "Black":  ([0, 0, 0], [180, 255, 50]),
    "White":  ([0, 0, 200], [180, 30, 255]),
}

class BloonDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bloon Detector")
        self.running = False
        self.check_vars = {}
        self.dartling_btn_pos = None
        self.pause_detection = False
        self.detection_area = None
        self.frame_bgr = None
        self.detected_boxes = []
        self.last_spawn_time = 0
        self.spawn_interval = 60  # seconds

        # GUI Elements
        ttk.Label(root, text="Detect Colors:").grid(row=0, column=0, sticky="w")
        for i, color in enumerate(COLOR_RANGES):
            var = BooleanVar(value=True)
            chk = ttk.Checkbutton(root, text=color, variable=var)
            chk.grid(row=i + 1, column=0, sticky="w")
            self.check_vars[color] = var

        self.start_button = ttk.Button(root, text="Start Detection", command=self.toggle_detection)
        self.start_button.grid(row=0, column=1, padx=10)

        self.log = scrolledtext.ScrolledText(root, width=40, height=15, state="disabled")
        self.log.grid(row=1, column=1, rowspan=10, padx=10, pady=5)

        self.calibrate_btn = ttk.Button(root, text="Calibrate Dartling Button", command=self.start_calibration)
        self.calibrate_btn.grid(row=12, column=0, pady=5)

        self.dartling_pos_label = ttk.Label(root, text="Dartling Button Pos: Not set")
        self.dartling_pos_label.grid(row=12, column=1, pady=5)

        self.sandbox_var = BooleanVar(value=False)
        self.sandbox_checkbox = ttk.Checkbutton(root, text="Enable Sandbox Mode", variable=self.sandbox_var)
        self.sandbox_checkbox.grid(row=13, column=0, sticky="w", pady=5)

        ttk.Label(root, text="Calibrate Detection Area:").grid(row=14, column=0, sticky="w")
        self.topleft = None
        self.bottomright = None
        ttk.Button(root, text="Set Top-Left", command=self.set_top_left).grid(row=15, column=0, sticky="w")
        ttk.Button(root, text="Set Bottom-Right", command=self.set_bottom_right).grid(row=16, column=0, sticky="w")
        self.area_label = ttk.Label(root, text="Detection Area: Default (center 800x600)")
        self.area_label.grid(row=17, column=0, columnspan=2, sticky="w", pady=5)

        # Live video feed label
        self.video_label = ttk.Label(root)
        self.video_label.grid(row=0, column=2, rowspan=18, padx=10, pady=5)

    def log_message(self, msg):
        self.log.configure(state="normal")
        self.log.insert(tk.END, msg + "\n")
        self.log.configure(state="disabled")
        self.log.yview(tk.END)

    def start_calibration(self):
        pos = pyautogui.position()
        self.dartling_btn_pos = (pos.x, pos.y)
        self.log_message(f"[INFO] Dartling Gun button calibrated at {self.dartling_btn_pos}")
        self.dartling_pos_label.config(text=f"Dartling Button Pos: {self.dartling_btn_pos}")

    def set_top_left(self):
        pos = pyautogui.position()
        self.topleft = pos
        self.log_message(f"[INFO] Detection area top-left set at {pos}")
        self.update_detection_area_label()

    def set_bottom_right(self):
        pos = pyautogui.position()
        self.bottomright = pos
        self.log_message(f"[INFO] Detection area bottom-right set at {pos}")
        self.update_detection_area_label()

    def update_detection_area_label(self):
        if self.topleft and self.bottomright:
            left = min(self.topleft.x, self.bottomright.x)
            top = min(self.topleft.y, self.bottomright.y)
            right = max(self.topleft.x, self.bottomright.x)
            bottom = max(self.topleft.y, self.bottomright.y)
            width = right - left
            height = bottom - top
            self.detection_area = {"top": top, "left": left, "width": width, "height": height}
            self.area_label.config(text=f"Detection Area: ({left}, {top}), {width}x{height}")
            self.log_message(f"[INFO] Detection area set to top-left {left},{top} size {width}x{height}")
        else:
            self.area_label.config(text="Detection Area: Incomplete (set both points)")

    def toggle_detection(self):
        if not self.running:
            self.running = True
            self.start_button.config(text="Stop Detection")
            threading.Thread(target=self.run_detection, daemon=True).start()
            threading.Thread(target=self.run_sandbox_mode, daemon=True).start()
            self.update_tkinter_view()
        else:
            self.running = False
            self.start_button.config(text="Start Detection")
            self.log_message("[INFO] Stopped detection.")

    def random_point_in_contour(self, contour):
        x, y, w, h = cv2.boundingRect(contour)
        for _ in range(100):
            px = random.randint(x, x + w)
            py = random.randint(y, y + h)
            if cv2.pointPolygonTest(contour, (px, py), False) >= 0:
                return (px, py)
        return None

    def spawn_dartling_random(self, green_mask, times=1):
        if not self.dartling_btn_pos:
            self.log_message("[ERROR] Dartling button position not calibrated.")
            return

        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_points = [self.random_point_in_contour(cnt) for cnt in contours if self.random_point_in_contour(cnt)]

        if not valid_points:
            self.log_message("[WARN] No valid green spots found for spawning.")
            return

        self.log_message("[INFO] Pausing detection for tower placement...")
        self.pause_detection = True

        offset_left = self.detection_area["left"] if self.detection_area else 0
        offset_top = self.detection_area["top"] if self.detection_area else 0

        for _ in range(times):
            spot = random.choice(valid_points)
            drop_x = spot[0] + offset_left
            drop_y = spot[1] + offset_top
            pyautogui.click(self.dartling_btn_pos[0], self.dartling_btn_pos[1])
            time.sleep(0.3)
            pyautogui.click(drop_x, drop_y)
            time.sleep(0.3)
            self.log_message(f"[INFO] Spawned Dartling Gun at {drop_x}, {drop_y}")

        self.pause_detection = False
        self.log_message("[INFO] Resumed detection after tower placement.")

    def run_sandbox_mode(self):
        keys = ['1', '2', '3', '4']
        while True:
            if not self.running:
                break
            if self.sandbox_var.get():
                for key in keys:
                    pyautogui.press(key)
                time.sleep(0.1)
            else:
                time.sleep(0.2)

    def run_detection(self):
        screen_width, screen_height = pyautogui.size()

        if self.detection_area:
            monitor = self.detection_area
        else:
            region_width, region_height = 800, 600
            top = (screen_height - region_height) // 2
            left = (screen_width - region_width) // 2
            monitor = {"top": top, "left": left, "width": region_width, "height": region_height}

        prev_frame = None
        self.log_message("[INFO] Starting detection...")

        with mss.mss() as sct:
            while self.running:
                if self.pause_detection:
                    time.sleep(0.1)
                    continue

                frame = np.array(sct.grab(monitor))
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                self.frame_bgr = frame_bgr.copy()

                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)

                if prev_frame is None:
                    prev_frame = gray
                    continue

                frame_diff = cv2.absdiff(prev_frame, gray)
                prev_frame = gray
                _, motion_mask = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
                motion_mask = cv2.dilate(motion_mask, None, iterations=2)

                hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
                targets = []
                self.detected_boxes.clear()

                for color, (lower, upper) in COLOR_RANGES.items():
                    if not self.check_vars[color].get():
                        continue
                    lower_np = np.array(lower, dtype=np.uint8)
                    upper_np = np.array(upper, dtype=np.uint8)
                    color_mask = cv2.inRange(hsv, lower_np, upper_np)
                    combined_mask = cv2.bitwise_and(motion_mask, color_mask)

                    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        if cv2.contourArea(cnt) > 300:
                            x, y, w, h = cv2.boundingRect(cnt)
                            center_x = monitor["left"] + x + w // 2
                            center_y = monitor["top"] + y + h // 2
                            targets.append((center_x, center_y))
                            self.detected_boxes.append((x, y, w, h, color))

                if targets:
                    try:
                        pyautogui.moveTo(targets[0][0], targets[0][1])
                        self.log_message(f"[INFO] Target at: {targets[0]}")
                    except Exception as e:
                        self.log_message(f"[ERROR] Mouse move failed: {e}")

                current_time = time.time()
                if current_time - self.last_spawn_time > self.spawn_interval:
                    green_lower, green_upper = COLOR_RANGES["Green"]
                    green_mask = cv2.inRange(hsv, np.array(green_lower), np.array(green_upper))
                    self.spawn_dartling_random(green_mask)
                    self.last_spawn_time = current_time

                time.sleep(0.05)

    def update_tkinter_view(self):
        if not self.running:
            return

        if self.frame_bgr is not None:
            frame_rgb = cv2.cvtColor(self.frame_bgr, cv2.COLOR_BGR2RGB)

            for (x, y, w, h, color) in self.detected_boxes:
                cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (255, 255, 255), 2)
                cv2.putText(frame_rgb, color, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            image = Image.fromarray(frame_rgb)
            image = image.resize((400, 300))
            photo = ImageTk.PhotoImage(image)

            self.video_label.config(image=photo)
            self.video_label.image = photo

        self.root.after(33, self.update_tkinter_view)

if __name__ == "__main__":
    root = tk.Tk()
    app = BloonDetectorApp(root)
    root.mainloop()
