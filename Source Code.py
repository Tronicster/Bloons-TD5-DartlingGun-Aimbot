import cv2
import numpy as np
import mss
import pyautogui
import threading
import time
import tkinter as tk
from tkinter import ttk, scrolledtext, BooleanVar
from PIL import Image, ImageTk
import logging
import json
import os
import torch
import torch.nn as nn
from torchvision import transforms
import serial # NEW: Import pyserial for serial communication
import serial.tools.list_ports # NEW: To list available serial ports

# --- Model definition ---
CLASS_NAMES = ["negative", "Red", "Blue", "Green", "Yellow", "Pink"]

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)

# --- Constants and Config ---
COLOR_RANGES = {
    "Blue": ([90, 80, 100], [140, 255, 255]),
    "Red": ([0, 120, 100], [10, 255, 255]),  # Red can also be detected with a second range for upper reds if needed
    "Green": ([50, 150, 50], [70, 255, 255]),
    "Yellow": ([22, 180, 180], [32, 255, 255]),
    "Pink": ([145, 100, 100], [170, 255, 255]),
}

SETTINGS_FILE = "bloon_detector_settings.json"
MODEL_PATH = "bloon_classifier.pth"  # Make sure this model file exists in the same directory

# --- Device setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load model ---
try:
    model = SimpleCNN(num_classes=len(CLASS_NAMES))
    # Load model state dictionary, specifying map_location to handle different devices
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)  # Move model to the detected device (CPU or GPU)
    model.eval()  # Set model to evaluation mode (important for inference)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file '{MODEL_PATH}' not found. Please ensure it's in the same directory.")
    exit()  # Exit the application if the model is essential for its functionality
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- Input transform matching training ---
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128 pixels
    transforms.ToTensor(),  # Convert PIL Image or numpy.ndarray to FloatTensor and scale to [0, 1]
    # Normalize with mean and std deviation used during training
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- App class ---
class BloonDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bloon Detector (CNN Integrated)")
        self.running = False  # Flag to control the main detection loop
        self.pause_detection = False  # Flag to pause detection without stopping the thread
        self.lock = threading.Lock()  # Lock for thread-safe access to shared variables
        self.detection_area = None  # Dictionary: {"top": int, "left": int, "width": int, "height": int}
        self.frame_bgr = None  # Stores the last processed BGR frame for display (with detections drawn)
        self.detected_boxes = []  # List of detected bloon info: [{"rect": (x,y,w,h), "center": (cx,cy), "color": str}]

        # Flags for control from UI
        self.spawn_enabled = BooleanVar(value=True)  # Controls if Dartling spawning is active
        self.upgrade_enabled = BooleanVar(value=True)  # Placeholder for future upgrade logic
        self.round_mode_enabled = BooleanVar(value=False)  # Controls auto-pressing space for rounds
        self.sandbox_var = BooleanVar(value=False)  # Placeholder for sandbox mode (if needed)
        self.hover_enabled = BooleanVar(value=True)  # New: Controls continuous mouse hovering

        self.topleft = None  # Stores (x, y) for top-left calibration point
        self.bottomright = None  # Stores (x, y) for bottom-right calibration point

        # Cooldown variables
        self.last_spawn_time = 0  # Timestamp of the last Dartling Gunner spawn
        self.last_hover_time = 0  # Timestamp of the last mouse hover action
        self.last_serial_send_time = 0 # NEW: Timestamp for last serial send
        self.serial_send_cooldown = 0.1 # MODIFIED: Cooldown for serial messages (e.g., 0.1 seconds)

        # NEW: Serial Port variables
        self.ser = None # Serial port object
        self.selected_com_port = tk.StringVar(value="COM5") # Default COM port
        self.selected_baud_rate = tk.StringVar(value="9600") # Default baud rate


        # --- UI Setup ---
        # Color detection checkboxes
        ttk.Label(root, text="Detect Colors:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.check_vars = {}
        for i, color in enumerate(COLOR_RANGES):
            var = BooleanVar(value=True)  # Default to true for all colors
            ttk.Checkbutton(root, text=color, variable=var).grid(row=i + 1, column=0, sticky="w", padx=15)
            self.check_vars[color] = var

        # Main control buttons
        self.start_button = ttk.Button(root, text="Start Detection", command=self.toggle_detection)
        self.start_button.grid(row=0, column=1, padx=10, pady=5)

        # Game mode specific controls
        self.round_mode_button = ttk.Checkbutton(root, text="Round Mode (Auto Press Space)",
                                                 variable=self.round_mode_enabled)
        self.round_mode_button.grid(row=1, column=1, sticky="w", padx=10, pady=5)

        # Log display area
        self.log = scrolledtext.ScrolledText(root, width=50, height=15, state="disabled")
        self.log.grid(row=2, column=1, rowspan=10, padx=10, pady=5)

        # Sandbox mode checkbox (currently unused in logic but present in UI)
        ttk.Checkbutton(root, text="Enable Sandbox Mode", variable=self.sandbox_var).grid(row=13, column=0, sticky="w",
                                                                                          pady=5, padx=5)

        # Calibration section
        ttk.Label(root, text="Calibrate Detection Area:").grid(row=14, column=0, sticky="w", padx=5)
        ttk.Button(root, text="Set Top-Left", command=self.set_top_left).grid(row=15, column=0, padx=15, sticky="w")
        ttk.Button(root, text="Set Bottom-Right", command=self.set_bottom_right).grid(row=16, column=0, padx=15,
                                                                                      sticky="w")

        self.area_label = ttk.Label(root, text="Detection Area: Not Set")
        self.area_label.grid(row=17, column=0, columnspan=2, sticky="w", pady=5, padx=5)

        # Feature toggles
        ttk.Checkbutton(root, text="Enable Dartling Spawning (Hotkey 'J')", variable=self.spawn_enabled).grid(row=18,
                                                                                                              column=0,
                                                                                                              sticky="w",
                                                                                                              padx=5,
                                                                                                              pady=5)
        ttk.Checkbutton(root, text="Enable Dartling Upgrading", variable=self.upgrade_enabled).grid(row=19, column=0,
                                                                                                    sticky="w", padx=5,
                                                                                                    pady=5)
        # New Hover Checkbox
        ttk.Checkbutton(root, text="Enable Bloon Hovering", variable=self.hover_enabled).grid(row=20, column=0,
                                                                                              sticky="w", padx=5,
                                                                                              pady=5)

        # NEW: Serial Port Configuration UI
        serial_frame = ttk.LabelFrame(root, text="Serial Port Settings")
        serial_frame.grid(row=21, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        ttk.Label(serial_frame, text="COM Port:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.com_port_dropdown = ttk.Combobox(serial_frame, textvariable=self.selected_com_port, state="readonly")
        self.com_port_dropdown.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        self.refresh_com_ports() # Populate dropdown on startup
        ttk.Button(serial_frame, text="Refresh Ports", command=self.refresh_com_ports).grid(row=0, column=2, padx=5, pady=2)


        ttk.Label(serial_frame, text="Baud Rate:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.baud_rate_dropdown = ttk.Combobox(serial_frame, textvariable=self.selected_baud_rate,
                                               values=["9600", "19200", "38400", "57600", "115200"], state="readonly")
        self.baud_rate_dropdown.grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        self.baud_rate_dropdown.set("9600") # Default

        self.connect_serial_button = ttk.Button(serial_frame, text="Connect Serial", command=self.toggle_serial_connection)
        self.connect_serial_button.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky="ew")


        # Video feed display
        self.video_label = ttk.Label(root)
        self.video_label.grid(row=0, column=2, rowspan=22, padx=10, pady=5)  # Adjusted rowspan for new serial section

        # Load settings from file on startup
        self.load_settings()
        if self.detection_area:
            left = self.detection_area["left"]
            top = self.detection_area["top"]
            width = self.detection_area["width"]
            height = self.detection_area["height"]
            self.area_label.config(text=f"Detection Area: ({left}, {top}), {width}x{height}")
        else:
            # Set a default detection area if none is loaded or set
            screen_width, screen_height = pyautogui.size()
            default_width = min(800, screen_width)
            default_height = min(600, screen_height)
            default_left = (screen_width - default_width) // 2
            default_top = (screen_height - default_height) // 2
            self.detection_area = {"top": default_top, "left": default_left,
                                   "width": default_width, "height": default_height}
            self.area_label.config(
                text=f"Detection Area: Default ({default_left}, {default_top}), {default_width}x{default_height}")
            self.log_message("No detection area loaded. Using default center screen area.")

        # Bind hotkey for manual spawn (J key)
        root.bind('<j>', self.spawn_helicopter_hotkey)

        # NEW: Handle window closing to ensure serial port is closed
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        """Handler for when the Tkinter window is closed."""
        self.running = False  # Stop the detection loop
        self.disconnect_serial_port() # Disconnect serial port
        self.save_settings() # Save settings on close
        self.root.destroy() # Destroy the Tkinter window

    def log_message(self, msg, level="info"):
        """Logs messages to the Tkinter text area and console."""
        self.log.configure(state="normal")
        self.log.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        self.log.configure(state="disabled")
        self.log.yview(tk.END)  # Auto-scroll to the end
        if level == "info":
            logging.info(msg)
        elif level == "error":
            logging.error(msg)
        elif level == "warning":
            logging.warning(msg)

    def set_top_left(self):
        """Initiates the process to capture the top-left coordinate after a delay."""
        self.log_message("Place cursor on top-left of detection area, waiting 3 seconds...")
        self.root.after(3000, self.capture_top_left)

    def capture_top_left(self):
        """Captures the current mouse position as the top-left coordinate."""
        self.topleft = pyautogui.position()
        self.log_message(f"Top-left set: {self.topleft}")
        self.update_detection_area_label()

    def set_bottom_right(self):
        """Initiates the process to capture the bottom-right coordinate after a delay."""
        self.log_message("Place cursor on bottom-right of detection area, waiting 3 seconds...")
        self.root.after(3000, self.capture_bottom_right)

    def capture_bottom_right(self):
        """Captures the current mouse position as the bottom-right coordinate."""
        self.bottomright = pyautogui.position()
        self.log_message(f"Bottom-right set: {self.bottomright}")
        self.update_detection_area_label()

    def update_detection_area_label(self):
        """Calculates and updates the detection area based on captured top-left and bottom-right points."""
        if self.topleft and self.bottomright:
            # Ensure coordinates are in correct order (min for top/left, abs diff for width/height)
            left = min(self.topleft.x, self.bottomright.x)
            top = min(self.topleft.y, self.bottomright.y)
            width = abs(self.topleft.x - self.bottomright.x)
            height = abs(self.topleft.y - self.bottomright.y)

            # Ensure width and height are at least 1 to avoid errors
            if width == 0: width = 1
            if height == 0: height = 1

            with self.lock:  # Use lock when modifying shared state
                self.detection_area = {"top": top, "left": left, "width": width, "height": height}
            self.area_label.config(text=f"Detection Area: ({left}, {top}), {width}x{height}")
            self.save_settings()
        else:
            self.log_message("Both top-left and bottom-right points needed to define area.", "warning")

    def save_settings(self):
        """Saves current settings (detection area, spawn/upgrade enabled, color checks) to a JSON file."""
        cfg = {
            "detection_area": self.detection_area,
            "spawn_enabled": self.spawn_enabled.get(),
            "upgrade_enabled": self.upgrade_enabled.get(),
            "round_mode_enabled": self.round_mode_enabled.get(),
            "sandbox_var": self.sandbox_var.get(),
            "hover_enabled": self.hover_enabled.get(),  # Save the new hover setting
            "color_checks": {c: var.get() for c, var in self.check_vars.items()},
            "serial_port": self.selected_com_port.get(), # NEW: Save selected serial port
            "baud_rate": self.selected_baud_rate.get() # NEW: Save selected baud rate
        }
        try:
            with open(SETTINGS_FILE, "w") as f:
                json.dump(cfg, f, indent=4)
            self.log_message("Settings saved successfully.")
        except Exception as e:
            self.log_message(f"Error saving settings: {e}", "error")

    def load_settings(self):
        """Loads settings from the JSON file."""
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r") as f:
                    cfg = json.load(f)
                self.detection_area = cfg.get("detection_area")
                self.spawn_enabled.set(cfg.get("spawn_enabled", True))
                self.upgrade_enabled.set(cfg.get("upgrade_enabled", True))
                self.round_mode_enabled.set(cfg.get("round_mode_enabled", False))
                self.sandbox_var.set(cfg.get("sandbox_var", False))
                self.hover_enabled.set(cfg.get("hover_enabled", True))  # Load the new hover setting, default True
                for c, v in cfg.get("color_checks", {}).items():
                    if c in self.check_vars:
                        self.check_vars[c].set(v)
                # NEW: Load serial port settings
                self.selected_com_port.set(cfg.get("serial_port", "COM5"))
                self.selected_baud_rate.set(cfg.get("baud_rate", "9600"))

                self.log_message("Settings loaded successfully.")
            except Exception as e:
                self.log_message(f"Failed loading settings: {e}. Using default values.", "error")
        else:
            self.log_message("Settings file not found. Using default application settings.", "info")

    # NEW: Serial Port Methods
    def refresh_com_ports(self):
        """Refreshes the list of available COM ports in the dropdown."""
        ports = serial.tools.list_ports.comports()
        port_names = [port.device for port in ports]
        self.com_port_dropdown['values'] = port_names
        if port_names and self.selected_com_port.get() not in port_names:
            self.selected_com_port.set(port_names[0]) # Set to first available if current isn't in list
        elif not port_names:
            self.selected_com_port.set("No Ports Found")

    def connect_serial_port(self):
        """Attempts to establish a serial connection."""
        if self.ser and self.ser.isOpen():
            self.log_message("Serial port is already open.", "warning")
            return

        port_name = self.selected_com_port.get()
        baud_rate = int(self.selected_baud_rate.get())

        if not port_name or port_name == "No Ports Found":
            self.log_message("No valid COM port selected.", "error")
            return

        try:
            self.ser = serial.Serial(port_name, baud_rate, timeout=1) # timeout can be adjusted
            self.log_message(f"Connected to {port_name} at {baud_rate} baud.", "info")
            self.connect_serial_button.config(text="Disconnect Serial", style="Danger.TButton")
        except serial.SerialException as e:
            self.log_message(f"Could not open serial port {port_name}: {e}", "error")
            self.ser = None
            self.connect_serial_button.config(text="Connect Serial", style="TButton")

    def disconnect_serial_port(self):
        """Closes the serial connection if open."""
        if self.ser and self.ser.isOpen():
            try:
                self.ser.close()
                self.log_message(f"Disconnected from {self.ser.port}.", "info")
            except Exception as e:
                self.log_message(f"Error closing serial port: {e}", "error")
            finally:
                self.ser = None
                self.connect_serial_button.config(text="Connect Serial", style="TButton")
        else:
            self.log_message("No active serial connection to disconnect.", "info")

    def toggle_serial_connection(self):
        """Toggles the serial connection on/off."""
        if self.ser and self.ser.isOpen():
            self.disconnect_serial_port()
        else:
            self.connect_serial_port()

    def send_bloon_counts_serial(self, bloon_counts):
        """Sends the bloon count data over the serial port with a cooldown."""
        if not self.ser or not self.ser.isOpen():
            return # Don't try to send if not connected

        current_time = time.time()
        if current_time - self.last_serial_send_time < self.serial_send_cooldown:
            return # Respect cooldown

        total_bloons = sum(bloon_counts.values())
        # Example format: "TotalBloons:5;Red:2;Blue:3\n"
        # You can adjust this format based on what your receiving device expects.
        detail_string = ";".join([f"{color}:{count}" for color, count in bloon_counts.items()])
        message = f"TotalBloons:{total_bloons};{detail_string}\n"

        try:
            self.ser.write(message.encode('utf-8')) # Encode string to bytes
            # self.log_message(f"Sent serial: {message.strip()}", "info") # Uncomment for verbose logging of serial sends
            self.last_serial_send_time = current_time
        except serial.SerialException as e:
            self.log_message(f"Error writing to serial port: {e}", "error")
            self.disconnect_serial_port() # Attempt to disconnect on error

    # End NEW: Serial Port Methods

    def toggle_detection(self):
        """Starts or stops the bloon detection thread."""
        if not self.running:
            if not self.detection_area:
                self.log_message("Please set the detection area first!", "warning")
                return
            self.running = True
            self.start_button.config(text="Stop Detection", style="Danger.TButton")  # Optional: change button style
            threading.Thread(target=self.run_detection, daemon=True).start()
            self.update_tkinter_view()  # Start updating the video feed
            self.log_message("Detection started.")
        else:
            self.running = False
            self.start_button.config(text="Start Detection", style="TButton")  # Optional: reset button style
            self.log_message("Detection stopped.")

    def run_detection(self):
        """Main detection loop running in a separate thread."""
        bg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=60, detectShadows=False)
        with mss.mss() as sct:
            while self.running:
                if self.pause_detection:
                    time.sleep(0.1)  # Briefly pause if detection is paused
                    continue

                # Ensure detection_area is valid before grabbing screenshot
                with self.lock:
                    current_monitor_area = self.detection_area.copy() if self.detection_area else None

                if not current_monitor_area:
                    self.log_message("Detection area is not set, pausing detection loop.", "warning")
                    time.sleep(1)  # Wait a bit before checking again
                    continue

                try:
                    # Grab screenshot of the defined detection area
                    img = np.array(sct.grab(current_monitor_area))
                    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Convert BGRA to BGR
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                    # Apply background subtraction
                    fg = bg.apply(frame, learningRate=0.01)
                    fg = cv2.erode(fg, None, iterations=1)
                    fg = cv2.dilate(fg, None, iterations=2)

                    # Create a mask for enabled colors
                    mask = np.zeros_like(fg)
                    for c, var in self.check_vars.items():
                        if var.get():  # Check if color detection is enabled for this color
                            if c in COLOR_RANGES:  # Ensure the color is defined in COLOR_RANGES
                                low, high = map(np.array, COLOR_RANGES.get(c, ([0, 0, 0], [180, 255, 255])))
                                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, low, high))
                            else:
                                self.log_message(f"Warning: Color '{c}' not found in COLOR_RANGES.", "warning")

                    # Combine motion detection (foreground mask) and color detection
                    final = cv2.bitwise_and(mask, fg)
                    cnts, _ = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    current_dets = []
                    bloon_counts_for_serial = {} # NEW: Dictionary to store bloon counts by color for serial
                    frame_with_detections = frame.copy()  # Create a copy to draw on
                    for c in cnts:
                        if cv2.contourArea(c) > 400:  # Filter small contours (adjust as needed)
                            x, y, w, h = cv2.boundingRect(c)
                            cropped = frame_with_detections.copy()[y:y + h,
                                      x:x + w]  # Crop the detected object from the frame copy

                            # Ensure cropped image is not empty before processing
                            if cropped.shape[0] == 0 or cropped.shape[1] == 0:
                                continue  # Skip empty crops

                            try:
                                # Prepare image for CNN model
                                pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                                tensor = transform(pil_img).unsqueeze(0).to(
                                    device)  # Add batch dimension and send to device

                                # Perform inference
                                with torch.no_grad():
                                    output = model(tensor)
                                    probs = torch.softmax(output, dim=1)  # Get probabilities

                                    # Get predicted class index and confidence
                                    pred_class_idx = torch.argmax(probs, dim=1).item()
                                    pred_confidence = probs[0, pred_class_idx].item()

                                if pred_class_idx != 0:  # Exclude 'negative' class (index 0)
                                    color_name = CLASS_NAMES[pred_class_idx]

                                    # Calculate absolute screen coordinates of the bloon's center
                                    cx = x + w // 2 + current_monitor_area["left"]
                                    cy = y + h // 2 + current_monitor_area["top"]
                                    current_dets.append({"rect": (x, y, w, h), "center": (cx, cy), "color": color_name})

                                    # NEW: Increment count for serial output
                                    bloon_counts_for_serial[color_name] = bloon_counts_for_serial.get(color_name, 0) + 1

                                    # --- Drawing on the frame for display ---
                                    # Draw rectangle on the frame for visual feedback (relative to frame)
                                    cv2.rectangle(frame_with_detections, (x, y), (x + w, y + h), (0, 255, 0), 2)

                                    # Get top 3 probabilities for display
                                    top_probs, top_indices = torch.topk(probs[0], k=min(3,
                                                                                        len(CLASS_NAMES)))  # Get top N, up to number of classes

                                    # Prepare probability text
                                    prob_text = f"{color_name}: {pred_confidence:.2f}"  # Start with the main prediction

                                    # Add top N probabilities if there are other significant ones
                                    detail_probs = []
                                    for i in range(top_probs.size(0)):
                                        class_idx = top_indices[i].item()
                                        if class_idx != pred_class_idx and CLASS_NAMES[class_idx] != "negative":
                                            detail_probs.append(f"{CLASS_NAMES[class_idx]}: {top_probs[i]:.2f}")

                                    if detail_probs:
                                        prob_text += " (" + ", ".join(detail_probs) + ")"

                                    # Draw predicted class and probabilities on the frame
                                    cv2.putText(frame_with_detections, color_name, (x, y - 25),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                    cv2.putText(frame_with_detections, prob_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                                0.4, (0, 200, 255), 1)  # Yellow-ish for probs

                            except Exception as e:
                                self.log_message(f"Model inference error for contour at ({x},{y}): {e}", "error")

                    with self.lock:  # Update shared state (frame and detected boxes)
                        self.frame_bgr = frame_with_detections  # Store the frame with detections
                        self.detected_boxes = current_dets  # Store detected objects with their info

                    # NEW: Send bloon counts over serial
                    if bloon_counts_for_serial: # Only send if some bloons were detected
                        self.send_bloon_counts_serial(bloon_counts_for_serial)
                    else: # If no bloons, send a message indicating zero bloons
                        # Send a dictionary with a single entry for total bloons = 0
                        self.send_bloon_counts_serial({"TotalBloons": 0})


                    # Check cooldown and spawn dartling gunner if enabled
                    self.spawn_dartgun()

                    # MODIFIED: Hover over detected bloons if enabled
                    self.hover_over_bloons()

                    # Auto-press space in Round Mode
                    if self.round_mode_enabled.get():
                        pyautogui.press("space")
                        time.sleep(0.1)  # Small delay to prevent spamming space

                except mss.exception.ScreenShotError as e:
                    self.log_message(f"Screen capture error: {e}. Check if detection area is valid.", "error")
                    self.running = False  # Stop detection on critical screen capture error
                    self.start_button.config(text="Start Detection", style="TButton")
                except Exception as e:
                    self.log_message(f"An unexpected error occurred in detection loop: {e}", "error")

                # Removed time.sleep(0.05) to allow for faster detection
                # time.sleep(0.05)

    def hover_over_bloons(self):
        """
        Moves the mouse cursor to hover over the first detected bloon,
        independent of spawning logic. Includes a cooldown.
        """
        HOVER_COOLDOWN_SECONDS = 0.01  # MODIFIED: Adjusted for faster hovering
        if not self.hover_enabled.get():
            return  # Do nothing if hovering is disabled

        if time.time() - self.last_hover_time < HOVER_COOLDOWN_SECONDS:
            return  # Respect cooldown

        with self.lock:
            # Get the center of the first detected bloon, if any
            if self.detected_boxes:
                target_x, target_y = self.detected_boxes[0]["center"]
                # MODIFIED: pyautogui.moveTo duration set to 0 for instantaneous movement
                pyautogui.moveTo(target_x, target_y, duration=0)
                self.last_hover_time = time.time()  # Update last hover time
            # else:
            # Optional: If no bloons are detected, you could move the mouse back to a neutral position
            # or simply leave it where it is. For now, we do nothing if no bloons are found.

    def spawn_dartgun(self):
        """
        Handles spawning of the Dartling Gunner, with cooldown and intelligent placement.
        Prioritizes detected bloons, falls back to a fixed position.
        """
        COOLDOWN_SECONDS = 5  # Can be adjusted
        if not self.spawn_enabled.get():
            return  # Do nothing if spawning is disabled

        if time.time() - self.last_spawn_time < COOLDOWN_SECONDS:
            # self.log_message("Dartling Gunner spawn on cooldown.", "info") # Uncomment for more verbose logging
            return

        target_x, target_y = -1, -1  # Initialize with invalid coordinates

        with self.lock:  # Acquire lock before accessing self.detected_boxes
            if self.detected_boxes:
                # Option 1: Move to the center of the first detected bloon
                # The 'center' key in detected_boxes already holds absolute screen coordinates
                bloon_center = self.detected_boxes[0]["center"]
                target_x, target_y = bloon_center[0], bloon_center[1]
                self.log_message(f"Moving to detected bloon at: ({target_x}, {target_y}) to spawn Dartling Gunner.")
            else:
                # Option 2: Move to a fixed, predefined location if no bloons are detected
                # IMPORTANT: Adjust these coordinates (x, y) to a suitable spot on your game map
                # You can use pyautogui.position() to find exact coordinates by hovering your mouse.
                # Example: Let's assume a central-lower part of the screen for placement
                screen_width, screen_height = pyautogui.size()
                target_x = screen_width // 2
                target_y = int(screen_height * 0.7)  # Approximately 70% down the screen
                self.log_message(
                    f"No bloons detected. Moving to default position: ({target_x}, {target_y}) to spawn Dartling Gunner.")

        if target_x != -1 and target_y != -1:  # Ensure coordinates are valid
            # Move the mouse smoothly to the target location
            pyautogui.moveTo(target_x, target_y, duration=0.2)  # Add a small duration for smoother movement

            # Click to select the spot or deselect existing tower/menu (common in games)
            pyautogui.click()
            time.sleep(0.1)  # Small pause after click

            # Press 'n' to select the Dartling Gunner (adjust key if different in your game)
            pyautogui.press("n")
            time.sleep(0.1)  # Small pause after key press

            # Click again to place the tower at the selected location
            pyautogui.click()
            self.log_message("Dartling Gunner spawned!", "info")

            # Update last spawn time to enforce cooldown
            self.last_spawn_time = time.time()
        else:
            self.log_message("Could not determine a valid target for spawning.", "warning")

    def update_tkinter_view(self):
        """Updates the Tkinter video feed label with the latest processed frame."""
        if not self.running:
            self.video_label.config(image=None)  # Clear the image when not running
            return
        with self.lock:
            frame = self.frame_bgr.copy() if self.frame_bgr is not None else None
        if frame is not None:
            # Resize frame to fit label if necessary, maintaining aspect ratio
            h, w, _ = frame.shape
            max_width = 640  # Max width for display
            max_height = 480  # Max height for display
            if w > max_width or h > max_height:
                scale_w = max_width / w
                scale_h = max_height / h
                scale = min(scale_w, scale_h)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk  # Keep a reference to prevent garbage collection
            self.video_label.config(image=imgtk)
        self.root.after(30, self.update_tkinter_view)  # Schedule next update

    def spawn_helicopter_hotkey(self, event):
        """Hot key handler for manually spawning Dartling Gunner (bound to 'j')."""
        self.log_message("Hotkey 'j' pressed.", "info")
        if self.spawn_enabled.get():
            self.spawn_dartgun()
        else:
            self.log_message("Dartling Spawning is disabled, cannot use hotkey.", "warning")


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    root = tk.Tk()
    # Configure a style for better button visuals
    style = ttk.Style()
    style.configure("TButton", font=("Inter", 10), padding=5, background="#e0e0e0", foreground="black")
    style.map("TButton", background=[("active", "#c0c0c0")])
    style.configure("Danger.TButton", background="#e74c3c", foreground="white")
    style.map("Danger.TButton", background=[("active", "#c0392b")])

    app = BloonDetectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
