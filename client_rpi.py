"""
Robot Vision Client - Raspberry Pi
====================================
Captures images, sends to FastAPI server, receives commands, controls motors.

Modes:
1. Real-time Tracking - Continuous single-image detection
2. Multi-view Finding - 3-view scan with servo control
3. Stream Mode - Server pulls from Pi's stream endpoint

Note: For stream mode, run rpi_stream_server.py first, then let the
      vision server pull frames from the stream.

Run: python client_rpi.py --server http://SERVER_IP:8000
"""

import io
import time
import argparse
import requests
import serial
import threading
import socket
import subprocess
import sys
from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass

import cv2
import numpy as np


# === Configuration ===

@dataclass
class Config:
    server_url: str = "http://localhost:8000"
    serial_port: str = "/dev/ttyUSB0"
    baudrate: int = 9600
    camera_width: int = 640
    camera_height: int = 480
    target_class: str = "person"
    confidence: float = 0.4
    mode: str = "realtime"  # "realtime" or "multiview"
    servo_settle_time: float = 0.5
    movement_duration: float = 1.5
    camera_index: int = 0  # USB camera index (0, 1, 2...)
    stream_port: int = 8080  # Port for local stream server
    use_stream: bool = True  # Use video stream for realtime mode


# === Motor Controller ===

class MotorController:
    """Controls motors via Arduino serial."""
    
    COMMANDS = {
        "far_left": "MFL",
        "left": "ML",
        "center": "MC",
        "right": "MR",
        "far_right": "MFR",
        "stop": "ST",
        "rotate_180": "RB",
        "servo_left": "SL",
        "servo_center": "SC",
        "servo_right": "SR",
    }
    
    def __init__(self, port: str, baudrate: int):
        self.ser = None
        self.connected = False
        self.rotation_ack = threading.Event()
        
        try:
            self.ser = serial.Serial(port=port, baudrate=baudrate, timeout=1)
            time.sleep(2)
            self.connected = True
            print(f"✓ Arduino connected on {port}")
            
            self._listener = threading.Thread(target=self._listen, daemon=True)
            self._listener.start()
        except Exception as e:
            print(f"⚠ Serial error: {e}")
            print("  Running in SIMULATION mode (no motors)")
    
    def _listen(self):
        """Listen for Arduino messages."""
        while True:
            if self.ser and self.ser.in_waiting:
                try:
                    line = self.ser.readline().decode('utf-8', errors='replace').strip()
                    if line:
                        print(f"  [Arduino] {line}")
                        if "ROTATION_ACK" in line:
                            self.rotation_ack.set()
                except Exception:
                    pass
            time.sleep(0.05)
    
    def send(self, cmd: str):
        """Send raw command."""
        print(f"  >> {cmd}")
        if self.ser:
            self.ser.write((cmd + "\n").encode('utf-8'))
            self.ser.flush()
    
    def move(self, direction: str, duration: float = 0):
        """Move in direction."""
        cmd = self.COMMANDS.get(direction, "ST")
        self.send(cmd)
        if duration > 0:
            time.sleep(duration)
            self.send("ST")
    
    def rotate_servo(self, position: str, wait_ack: bool = True, timeout: float = 3.0) -> bool:
        """Rotate servo to position."""
        cmd = self.COMMANDS.get(f"servo_{position}", "SC")
        self.send(cmd)
        
        if wait_ack and self.connected:
            if self.rotation_ack.wait(timeout):
                self.rotation_ack.clear()
                self.send("PHOTO_ACK")
                return True
            print("  ⚠ Rotation timeout")
            return False
        
        time.sleep(0.5)  # Fallback delay
        return True
    
    def rotate_180(self):
        """Rotate robot 180 degrees."""
        self.send("RB")
        time.sleep(3.0)
    
    def stop(self):
        self.send("ST")
    
    def close(self):
        if self.ser:
            self.ser.close()


# === Camera ===

class Camera:
    """Handles image capture."""
    
    def __init__(self, config: Config):
        self.config = config
        self.camera = None
        self._init_camera()
    
    def _init_camera(self):
        """Initialize USB webcam."""
        self.camera = cv2.VideoCapture(self.config.camera_index)
        if not self.camera.isOpened():
            print(f"Warning: Camera index {self.config.camera_index} failed, trying next...")
            self.camera = cv2.VideoCapture(self.config.camera_index + 1)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        print(f"✓ USB Webcam initialized (index {self.config.camera_index})")
        time.sleep(2)
    
    def capture(self) -> Optional[np.ndarray]:
        """Capture a frame from USB webcam."""
        ret, frame = self.camera.read()
        if not ret:
            return None
        return frame
    
    def capture_jpeg(self) -> Optional[bytes]:
        """Capture frame as JPEG bytes."""
        frame = self.capture()
        if frame is None:
            return None
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buffer.tobytes()
    
    def close(self):
        """Release USB webcam."""
        self.camera.release()


# === API Client ===

class VisionClient:
    """Communicates with the FastAPI server."""
    
    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip('/')
        self.session = requests.Session()
        self._check_connection()
    
    def _check_connection(self):
        """Check server connection."""
        try:
            resp = self.session.get(f"{self.server_url}/", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                print(f"✓ Connected to server: {data['status']}")
                print(f"  Model loaded: {data['model_loaded']}")
            else:
                print(f"⚠ Server returned status {resp.status_code}")
        except Exception as e:
            print(f"⚠ Cannot connect to server: {e}")
    
    def detect_realtime(self, image_bytes: bytes, target_class: str = "person",
                        confidence: float = 0.4, track: bool = True) -> Dict[str, Any]:
        """Send single image for real-time detection."""
        try:
            files = {"image": ("frame.jpg", image_bytes, "image/jpeg")}
            data = {
                "target_class": target_class,
                "confidence": confidence,
                "track": track
            }
            
            resp = self.session.post(
                f"{self.server_url}/detect/realtime",
                files=files,
                data=data,
                timeout=10
            )
            
            if resp.status_code == 200:
                return resp.json()
            else:
                return {"success": False, "error": resp.text}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def detect_multiview(self, images: Dict[str, bytes], target_class: str = "person",
                         confidence: float = 0.4) -> Dict[str, Any]:
        """Send 3 images for multi-view detection."""
        try:
            files = {
                "image_left": ("left.jpg", images["left"], "image/jpeg"),
                "image_center": ("center.jpg", images["center"], "image/jpeg"),
                "image_right": ("right.jpg", images["right"], "image/jpeg"),
            }
            data = {
                "target_class": target_class,
                "confidence": confidence
            }
            
            resp = self.session.post(
                f"{self.server_url}/detect/multiview",
                files=files,
                data=data,
                timeout=15
            )
            
            if resp.status_code == 200:
                return resp.json()
            else:
                return {"success": False, "error": resp.text}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def reset_tracker(self):
        """Reset server-side tracker."""
        try:
            resp = self.session.post(f"{self.server_url}/tracker/reset", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False
    
    def start_stream(self, stream_url: str, target_class: str = "person", 
                     confidence: float = 0.4, interval_ms: int = 100):
        """Start server-side stream processing."""
        try:
            data = {
                "stream_url": stream_url,
                "target_class": target_class,
                "confidence": confidence,
                "interval_ms": interval_ms
            }
            resp = self.session.post(f"{self.server_url}/stream/start", json=data, timeout=5)
            return resp.status_code == 200
        except Exception as e:
            print(f"Failed to start stream: {e}")
            return False
    
    def stop_stream(self):
        """Stop server-side stream processing."""
        try:
            resp = self.session.post(f"{self.server_url}/stream/stop", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False
    
    def detect_realtime_stream(self) -> Dict[str, Any]:
        """Get real-time detection result from stream."""
        try:
            resp = self.session.get(f"{self.server_url}/detect/realtime/stream", timeout=5)
            if resp.status_code == 200:
                return resp.json()
            else:
                return {"success": False, "error": resp.text}
        except Exception as e:
            return {"success": False, "error": str(e)}


# === Main Robot Controller ===

class RobotController:
    """Main robot control loop."""
    
    def __init__(self, config: Config):
        self.config = config
        self.camera = Camera(config)
        self.motors = MotorController(config.serial_port, config.baudrate)
        self.client = VisionClient(config.server_url)
        self.running = False
        self.stream_process = None
        self.stream_url = None
    
    def _get_local_ip(self) -> str:
        """Get local IP address."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "localhost"
    
    def _start_stream_server(self):
        """Start stream server in background process."""
        if self.stream_process:
            return
        
        try:
            # Start stream server as subprocess
            cmd = [
                sys.executable, "rpi_stream_server.py",
                "--port", str(self.config.stream_port),
                "--camera", str(self.config.camera_index),
                "--width", str(self.config.camera_width),
                "--height", str(self.config.camera_height)
            ]
            self.stream_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            time.sleep(3)  # Wait for server to start
            pi_ip = self._get_local_ip()
            self.stream_url = f"http://{pi_ip}:{self.config.stream_port}"
            print(f"✓ Stream server started at {self.stream_url}")
            return True
        except Exception as e:
            print(f"⚠ Failed to start stream server: {e}")
            return False
    
    def _stop_stream_server(self):
        """Stop stream server."""
        if self.stream_process:
            self.stream_process.terminate()
            self.stream_process.wait(timeout=5)
            self.stream_process = None
            print("✓ Stream server stopped")
    
    def run_realtime(self):
        """Run real-time tracking mode using video stream."""
        print("\n" + "=" * 50)
        print("  REAL-TIME TRACKING MODE (Video Stream)")
        print(f"  Target: {self.config.target_class}")
        print("=" * 50 + "\n")
        
        # Start stream server if using stream mode
        if self.config.use_stream:
            if not self._start_stream_server():
                print("⚠ Falling back to image-based mode")
                self.config.use_stream = False
        
        # Start server-side stream processing
        if self.config.use_stream and self.stream_url:
            print(f"Starting server stream processing from {self.stream_url}...")
            if not self.client.start_stream(
                stream_url=self.stream_url,
                target_class=self.config.target_class,
                confidence=self.config.confidence,
                interval_ms=100
            ):
                print("⚠ Failed to start server stream processing, falling back to image mode")
                self.config.use_stream = False
        
        self.running = True
        last_command_time = 0
        command_cooldown = 0.5
        
        try:
            while self.running:
                if self.config.use_stream:
                    # Get result from stream
                    result = self.client.detect_realtime_stream()
                else:
                    # Fallback: capture and send image
                    image_bytes = self.camera.capture_jpeg()
                    if image_bytes is None:
                        print("Failed to capture image")
                        continue
                    
                    result = self.client.detect_realtime(
                        image_bytes,
                        target_class=self.config.target_class,
                        confidence=self.config.confidence,
                        track=True
                    )
                
                if not result.get("success"):
                    print(f"Detection failed: {result.get('error')}")
                    time.sleep(0.1)
                    continue
                
                # Process result
                direction = result.get("direction", "none")
                target_found = result.get("target_found", False)
                reached = result.get("reached", False)
                
                if target_found:
                    tracking = result.get("tracking", {})
                    velocity = tracking.get("velocity", [0, 0]) if tracking else [0, 0]
                    area = result.get("distance_ratio", 0)
                    
                    print(f"Target: dir={direction}, area={area:.1%}, vel=({velocity[0]:.1f}, {velocity[1]:.1f})")
                    
                    if reached:
                        print("🎉 TARGET REACHED!")
                        self.motors.stop()
                    else:
                        # Smart movement with velocity prediction
                        current_time = time.time()
                        if current_time - last_command_time > command_cooldown:
                            if direction == "left":
                                if velocity[0] > 50:  # Moving right
                                    print("  Object moving to center, waiting...")
                                else:
                                    self.motors.move("left", self.config.movement_duration)
                            elif direction == "right":
                                if velocity[0] < -50:  # Moving left
                                    print("  Object moving to center, waiting...")
                                else:
                                    self.motors.move("right", self.config.movement_duration)
                            else:  # center
                                self.motors.move("center", self.config.movement_duration)
                            
                            last_command_time = current_time
                else:
                    print("Target not found - searching...")
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            if self.config.use_stream:
                self.client.stop_stream()
                self._stop_stream_server()
            self.motors.stop()
    
    def run_multiview(self):
        """Run multi-view object finding mode."""
        print("\n" + "=" * 50)
        print("  MULTI-VIEW OBJECT FINDING MODE")
        print(f"  Target: {self.config.target_class}")
        print("  5-Direction Navigation")
        print("=" * 50 + "\n")
        
        self.running = True
        
        try:
            while self.running:
                # === FRONT SCAN ===
                print("\n[FRONT SCAN]")
                images = self._capture_3_views()
                
                if not images:
                    print("Failed to capture images")
                    continue
                
                # Send to server
                result = self.client.detect_multiview(
                    images,
                    target_class=self.config.target_class,
                    confidence=self.config.confidence
                )
                
                if not result.get("success"):
                    print(f"Detection failed: {result.get('error')}")
                    continue
                
                # Process result
                direction = result.get("direction", "none")
                target_found = result.get("target_found", False)
                reached = result.get("reached", False)
                prominence = result.get("prominence", {})
                
                print(f"  Prominence: L={prominence.get('left', 0):.2f}, "
                      f"C={prominence.get('center', 0):.2f}, R={prominence.get('right', 0):.2f}")
                print(f"  Result: {result.get('message')}")
                
                if target_found:
                    if reached:
                        print("\n🎉 TARGET REACHED!")
                        self.motors.stop()
                        time.sleep(2)
                    else:
                        # Navigate
                        print(f"  Moving: {direction}")
                        self.motors.move(direction, self.config.movement_duration)
                else:
                    # === BACK SCAN ===
                    print("\n[ROTATING 180° - BACK SCAN]")
                    self.motors.rotate_180()
                    
                    images = self._capture_3_views()
                    if not images:
                        continue
                    
                    result = self.client.detect_multiview(
                        images,
                        target_class=self.config.target_class,
                        confidence=self.config.confidence
                    )
                    
                    if result.get("target_found"):
                        direction = result.get("direction", "center")
                        print(f"  Target found in back! Direction: {direction}")
                        self.motors.move(direction, self.config.movement_duration)
                    else:
                        print("  Target not found - searching...")
                        self.motors.move("center", 1.0)
                    
                    # Rotate back
                    self.motors.rotate_180()
                
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.motors.stop()
    
    def _capture_3_views(self) -> Optional[Dict[str, bytes]]:
        """Capture left, center, right views."""
        images = {}
        
        for view, servo_pos in [("center", "center"), ("left", "left"), ("right", "right")]:
            print(f"  Scanning {view}...")
            self.motors.rotate_servo(servo_pos)
            time.sleep(self.config.servo_settle_time)
            
            img_bytes = self.camera.capture_jpeg()
            if img_bytes is None:
                print(f"  Failed to capture {view}")
                return None
            
            images[view] = img_bytes
        
        # Return to center
        self.motors.rotate_servo("center")
        return images
    
    def run(self):
        """Run selected mode."""
        try:
            if self.config.mode == "realtime":
                self.run_realtime()
            else:
                self.run_multiview()
        finally:
            self.close()
    
    def close(self):
        """Cleanup."""
        if self.config.use_stream:
            self.client.stop_stream()
            self._stop_stream_server()
        self.motors.stop()
        self.camera.close()
        self.motors.close()
        print("Cleanup complete")


# === Entry Point ===

def main():
    parser = argparse.ArgumentParser(description="Robot Vision Client")
    parser.add_argument("--server", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Arduino serial port")
    parser.add_argument("--target", default="person", help="Target object class")
    parser.add_argument("--mode", choices=["realtime", "multiview"], default="realtime",
                        help="Detection mode")
    parser.add_argument("--confidence", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--camera", type=int, default=0, help="USB camera index (default: 0)")
    parser.add_argument("--stream-port", type=int, default=8080, help="Stream server port (default: 8080)")
    parser.add_argument("--no-stream", action="store_true", help="Disable video stream, use image-based mode")
    
    args = parser.parse_args()
    
    config = Config(
        server_url=args.server,
        serial_port=args.port,
        target_class=args.target,
        mode=args.mode,
        confidence=args.confidence,
        camera_index=args.camera,
        stream_port=args.stream_port,
        use_stream=not args.no_stream
    )
    
    robot = RobotController(config)
    robot.run()


if __name__ == "__main__":
    main()

