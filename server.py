"""
Robot Vision Server - FastAPI + YOLOv11n
==========================================
Accepts images from Raspberry Pi, processes with YOLO, returns navigation commands.

Modes:
1. Real-time Tracking - Single image → direction + tracking info
2. Multi-view Object Finding - 3 images (L/C/R) → 5-direction result
3. Stream Processing - Pull frames from Pi's stream endpoint

Run: uvicorn server:app --host 0.0.0.0 --port 8000
"""

import io
import time
import base64
import asyncio
import numpy as np
import socket
from enum import Enum
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from collections import deque
import threading

import cv2
import requests
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from ultralytics import YOLO
import torch

# === FastAPI App ===
app = FastAPI(
    title="Robot Vision Server",
    description="YOLOv11n Object Detection & Tracking API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === YOLO Model ===
# Detect and configure GPU device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("⚠ GPU not available, using CPU (slower)")

print(f"Loading YOLOv11n model on {device.upper()}...")
model = YOLO("C:\\Users\\sushu\\OneDrive\\Desktop\\Abiogensis_stero\\yolo\\yolo11n.pt")
# Move model to GPU if available
model.to(device)
CLASS_NAMES = model.names
print(f"✓ Model loaded! {len(CLASS_NAMES)} classes available")
print(f"  Device: {device.upper()}")

# Get local IP address for network access
def get_local_ip():
    """Get local IP address for network access."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None

local_ip = get_local_ip()

print("\n" + "=" * 60)
print("  SERVER STARTING")
print("=" * 60)
print("  Access the server at:")
print("    • http://localhost:8000/ (same machine)")
print("    • http://127.0.0.1:8000/ (same machine)")
if local_ip:
    print(f"    • http://{local_ip}:8000/ (from other devices on network)")
print("  API docs: http://localhost:8000/docs")
print("=" * 60 + "\n")


# === Enums ===

class Direction(str, Enum):
    FAR_LEFT = "far_left"
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    FAR_RIGHT = "far_right"
    NONE = "none"


class Mode(str, Enum):
    REALTIME = "realtime"
    MULTIVIEW = "multiview"
    STREAM = "stream"


# === Response Models ===

class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]
    centroid: List[int]  # [x, y]
    area: int
    area_ratio: float


class TrackingInfo(BaseModel):
    object_id: int
    velocity: List[float]  # [vx, vy] pixels/sec
    frames_tracked: int


class RealtimeResponse(BaseModel):
    success: bool
    direction: str
    target_found: bool
    target_detection: Optional[Detection] = None
    tracking: Optional[TrackingInfo] = None
    all_detections: List[Detection] = []
    distance_ratio: float = 0.0
    reached: bool = False
    message: str = ""


class MultiviewResponse(BaseModel):
    success: bool
    direction: str
    direction_angle: int
    target_found: bool
    views_with_target: List[str] = []
    prominence: Dict[str, float] = {}
    best_view: Optional[str] = None
    target_area_ratio: float = 0.0
    reached: bool = False
    obstacles_detected: int = 0
    message: str = ""


class StreamConfig(BaseModel):
    stream_url: str
    target_class: str = "person"
    confidence: float = 0.4
    interval_ms: int = 100


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    available_classes: List[str]
    stream_active: bool = False


# === Tracker State (Server-side) ===

class ServerTracker:
    """Maintains tracking state across requests from the same client."""
    
    def __init__(self):
        self.tracked_objects: Dict[int, deque] = {}
        self.next_id = 0
        self.max_distance = 100
        self.max_history = 30
        self.last_update = time.time()
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """Match new detections to existing tracks."""
        current_time = time.time()
        
        # Clear stale tracks (no update for 5 seconds)
        if current_time - self.last_update > 5.0:
            self.tracked_objects.clear()
            self.next_id = 0
        
        self.last_update = current_time
        matched = []
        unmatched_indices = list(range(len(detections)))
        
        # Match to existing tracks
        for obj_id, history in list(self.tracked_objects.items()):
            if not history:
                continue
            
            last_centroid = history[-1]["centroid"]
            best_match = None
            best_dist = float('inf')
            
            for idx in unmatched_indices:
                det = detections[idx]
                cx, cy = det["centroid"]
                dist = np.sqrt((cx - last_centroid[0])**2 + (cy - last_centroid[1])**2)
                
                if dist < best_dist and dist < self.max_distance:
                    best_dist = dist
                    best_match = idx
            
            if best_match is not None:
                det = detections[best_match]
                history.append({
                    "centroid": det["centroid"],
                    "time": current_time
                })
                det["object_id"] = obj_id
                det["frames_tracked"] = len(history)
                matched.append(det)
                unmatched_indices.remove(best_match)
            else:
                # Remove stale track
                if current_time - history[-1]["time"] > 1.0:
                    del self.tracked_objects[obj_id]
        
        # Create new tracks
        for idx in unmatched_indices:
            det = detections[idx]
            self.tracked_objects[self.next_id] = deque(maxlen=self.max_history)
            self.tracked_objects[self.next_id].append({
                "centroid": det["centroid"],
                "time": current_time
            })
            det["object_id"] = self.next_id
            det["frames_tracked"] = 1
            matched.append(det)
            self.next_id += 1
        
        return matched
    
    def get_velocity(self, object_id: int) -> List[float]:
        """Calculate velocity for tracked object."""
        if object_id not in self.tracked_objects:
            return [0.0, 0.0]
        
        history = self.tracked_objects[object_id]
        if len(history) < 2:
            return [0.0, 0.0]
        
        p1 = history[-2]
        p2 = history[-1]
        dt = p2["time"] - p1["time"]
        
        if dt == 0:
            return [0.0, 0.0]
        
        vx = (p2["centroid"][0] - p1["centroid"][0]) / dt
        vy = (p2["centroid"][1] - p1["centroid"][1]) / dt
        
        return [round(vx, 2), round(vy, 2)]


# === Stream Processor ===

class StreamProcessor:
    """Pulls frames from Pi's stream and processes them."""
    
    def __init__(self):
        self.running = False
        self.stream_url: Optional[str] = None
        self.target_class = "person"
        self.confidence = 0.4
        self.interval = 0.1  # seconds
        self.latest_result: Optional[Dict] = None
        self.tracker = ServerTracker()
        self._thread: Optional[threading.Thread] = None
    
    def start(self, config: StreamConfig):
        """Start processing stream."""
        if self.running:
            self.stop()
        
        self.stream_url = config.stream_url.rstrip('/') + "/api/frame"
        self.target_class = config.target_class
        self.confidence = config.confidence
        self.interval = config.interval_ms / 1000.0
        self.running = True
        
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        print(f"✓ Stream processor started: {self.stream_url}")
    
    def stop(self):
        """Stop processing."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self.latest_result = None
        print("Stream processor stopped")
    
    def _process_loop(self):
        """Background loop to fetch and process frames."""
        session = requests.Session()
        
        while self.running:
            try:
                # Fetch frame from Pi
                resp = session.get(self.stream_url, timeout=2)
                if resp.status_code != 200:
                    time.sleep(self.interval)
                    continue
                
                # Decode image
                nparr = np.frombuffer(resp.content, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    continue
                
                # Run detection
                height, width = img.shape[:2]
                frame_area = width * height
                
                results = model(img, device=device, verbose=False)[0]
                detections = []
                
                for box in results.boxes:
                    class_id = int(box.cls[0])
                    class_name = CLASS_NAMES[class_id]
                    confidence = float(box.conf[0])
                    
                    if confidence < self.confidence:
                        continue
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)
                    
                    detections.append({
                        "class_name": class_name,
                        "confidence": round(confidence, 3),
                        "bbox": [x1, y1, x2, y2],
                        "centroid": [(x1 + x2) // 2, (y1 + y2) // 2],
                        "area": area,
                        "area_ratio": round(area / frame_area, 4),
                        "is_target": class_name == self.target_class
                    })
                
                # Filter and track targets
                target_dets = [d for d in detections if d["is_target"]]
                if target_dets:
                    target_dets = self.tracker.update(target_dets)
                
                # Find best target
                best_target = None
                if target_dets:
                    best_target = max(target_dets, key=lambda d: d["area"])
                
                # Calculate direction
                direction = "none"
                area_ratio = 0.0
                reached = False
                tracking_info = None
                
                if best_target:
                    cx = best_target["centroid"][0]
                    if cx < width * 0.35:
                        direction = "left"
                    elif cx > width * 0.65:
                        direction = "right"
                    else:
                        direction = "center"
                    
                    area_ratio = best_target["area_ratio"]
                    reached = area_ratio > 0.25
                    
                    # Get tracking info with velocity
                    if "object_id" in best_target:
                        velocity = self.tracker.get_velocity(best_target["object_id"])
                        tracking_info = {
                            "object_id": best_target["object_id"],
                            "velocity": velocity,
                            "frames_tracked": best_target.get("frames_tracked", 1)
                        }
                
                # Store result with full tracking info
                self.latest_result = {
                    "timestamp": time.time(),
                    "direction": direction,
                    "target_found": best_target is not None,
                    "target": best_target,
                    "all_detections": detections,
                    "frame_size": [width, height],
                    "area_ratio": area_ratio,
                    "reached": reached,
                    "tracking": tracking_info
                }
                
                time.sleep(self.interval)
                
            except Exception as e:
                print(f"Stream error: {e}")
                time.sleep(1.0)
    
    def get_latest(self) -> Optional[Dict]:
        """Get latest processing result."""
        return self.latest_result


# Global instances
tracker = ServerTracker()
stream_processor = StreamProcessor()


# === Helper Functions ===

def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes to numpy array."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def run_detection(image: np.ndarray, target_class: Optional[str] = None, 
                  confidence_threshold: float = 0.4) -> List[Dict]:
    """Run YOLO detection on image."""
    results = model(image, device=device, verbose=False)[0]
    detections = []
    
    height, width = image.shape[:2]
    frame_area = width * height
    
    for box in results.boxes:
        class_id = int(box.cls[0])
        class_name = CLASS_NAMES[class_id]
        confidence = float(box.conf[0])
        
        if confidence < confidence_threshold:
            continue
        
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)
        
        det = {
            "class_name": class_name,
            "confidence": round(confidence, 3),
            "bbox": [x1, y1, x2, y2],
            "centroid": [(x1 + x2) // 2, (y1 + y2) // 2],
            "area": area,
            "area_ratio": round(area / frame_area, 4),
            "is_target": target_class is None or class_name == target_class
        }
        detections.append(det)
    
    return detections


def get_zone_direction(centroid_x: int, width: int) -> str:
    """Get 3-zone direction (left/center/right)."""
    left_bound = int(width * 0.35)
    right_bound = int(width * 0.65)
    
    if centroid_x < left_bound:
        return "left"
    elif centroid_x > right_bound:
        return "right"
    return "center"


def calculate_5_direction(prominences: Dict[str, float]) -> Direction:
    """Calculate 5-direction from view prominences."""
    p_left = prominences.get("left", 0)
    p_center = prominences.get("center", 0)
    p_right = prominences.get("right", 0)
    
    total = p_left + p_center + p_right
    if total == 0:
        return Direction.NONE
    
    # Normalize
    p_left /= total
    p_center /= total
    p_right /= total
    
    # Weighted position: left=-1, center=0, right=1
    weighted = -1 * p_left + 0 * p_center + 1 * p_right
    
    if weighted < -0.6:
        return Direction.FAR_LEFT
    elif weighted < -0.2:
        return Direction.LEFT
    elif weighted < 0.2:
        return Direction.CENTER
    elif weighted < 0.6:
        return Direction.RIGHT
    else:
        return Direction.FAR_RIGHT


# === API Endpoints ===

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="online",
        model_loaded=True,
        available_classes=list(CLASS_NAMES.values())[:30],
        stream_active=stream_processor.running
    )


@app.get("/classes")
async def get_classes():
    """Get all available COCO classes."""
    return {"classes": list(CLASS_NAMES.values())}


@app.post("/detect/realtime", response_model=RealtimeResponse)
async def detect_realtime(
    image: UploadFile = File(...),
    target_class: str = Form("person"),
    confidence: float = Form(0.4),
    track: bool = Form(True)
):
    """
    Real-time single image detection and tracking.
    
    - **image**: Image file (JPEG/PNG)
    - **target_class**: COCO class to track (default: person)
    - **confidence**: Detection confidence threshold (default: 0.4)
    - **track**: Enable tracking across frames (default: True)
    
    Returns direction (left/center/right) and tracking info.
    """
    try:
        # Read and decode image
        contents = await image.read()
        img = decode_image(contents)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        height, width = img.shape[:2]
        
        # Run detection
        all_detections = run_detection(img, None, confidence)
        target_detections = [d for d in all_detections if d["class_name"] == target_class]
        
        # Update tracker if enabled
        if track and target_detections:
            target_detections = tracker.update(target_detections)
        
        # Find best target (largest)
        best_target = None
        if target_detections:
            best_target = max(target_detections, key=lambda d: d["area"])
        
        # Determine direction and response
        if best_target:
            direction = get_zone_direction(best_target["centroid"][0], width)
            area_ratio = best_target["area_ratio"]
            reached = area_ratio > 0.25
            
            # Get tracking info
            tracking_info = None
            if track and "object_id" in best_target:
                velocity = tracker.get_velocity(best_target["object_id"])
                tracking_info = TrackingInfo(
                    object_id=best_target["object_id"],
                    velocity=velocity,
                    frames_tracked=best_target.get("frames_tracked", 1)
                )
            
            return RealtimeResponse(
                success=True,
                direction=direction,
                target_found=True,
                target_detection=Detection(
                    class_name=best_target["class_name"],
                    confidence=best_target["confidence"],
                    bbox=best_target["bbox"],
                    centroid=best_target["centroid"],
                    area=best_target["area"],
                    area_ratio=best_target["area_ratio"]
                ),
                tracking=tracking_info,
                all_detections=[Detection(**{k: v for k, v in d.items() 
                                            if k in Detection.__fields__}) 
                               for d in all_detections],
                distance_ratio=area_ratio,
                reached=reached,
                message=f"Target '{target_class}' found - {direction}" + 
                        (" - REACHED!" if reached else "")
            )
        else:
            return RealtimeResponse(
                success=True,
                direction="none",
                target_found=False,
                all_detections=[Detection(**{k: v for k, v in d.items() 
                                            if k in Detection.__fields__}) 
                               for d in all_detections],
                message=f"Target '{target_class}' not found"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/multiview", response_model=MultiviewResponse)
async def detect_multiview(
    image_left: UploadFile = File(...),
    image_center: UploadFile = File(...),
    image_right: UploadFile = File(...),
    target_class: str = Form("person"),
    confidence: float = Form(0.4)
):
    """
    Multi-view object finding with 5-direction output.
    
    - **image_left**: Left view image
    - **image_center**: Center view image  
    - **image_right**: Right view image
    - **target_class**: COCO class to find (default: person)
    - **confidence**: Detection confidence threshold (default: 0.4)
    
    Returns one of 5 directions: far_left, left, center, right, far_right
    """
    try:
        # Read all images
        images = {}
        for name, file in [("left", image_left), ("center", image_center), ("right", image_right)]:
            contents = await file.read()
            img = decode_image(contents)
            if img is None:
                raise HTTPException(status_code=400, detail=f"Invalid {name} image")
            images[name] = img
        
        # Detect in each view
        view_results = {}
        prominences = {}
        obstacles_count = 0
        
        for view_name, img in images.items():
            height, width = img.shape[:2]
            frame_area = width * height
            
            all_dets = run_detection(img, None, confidence)
            target_dets = [d for d in all_dets if d["class_name"] == target_class]
            
            # Count obstacles (non-target large objects)
            obstacles = [d for d in all_dets 
                        if d["class_name"] != target_class and d["area_ratio"] > 0.05]
            obstacles_count += len(obstacles)
            
            # Calculate prominence (how prominent is target in this view)
            if target_dets:
                best = max(target_dets, key=lambda d: d["area"])
                # Scale area ratio to prominence (0-1)
                prominences[view_name] = min(1.0, best["area_ratio"] * 5)
            else:
                prominences[view_name] = 0.0
            
            view_results[view_name] = {
                "target_dets": target_dets,
                "all_dets": all_dets
            }
        
        # Calculate 5-direction
        direction = calculate_5_direction(prominences)
        direction_angles = {
            Direction.FAR_LEFT: 0,
            Direction.LEFT: 45,
            Direction.CENTER: 90,
            Direction.RIGHT: 135,
            Direction.FAR_RIGHT: 180,
            Direction.NONE: -1
        }
        
        # Find best view
        views_with_target = [v for v, p in prominences.items() if p > 0]
        best_view = max(prominences, key=prominences.get) if views_with_target else None
        
        # Check if reached
        max_area = max((p for p in prominences.values()), default=0)
        reached = max_area > 0.5  # Very prominent = close
        
        return MultiviewResponse(
            success=True,
            direction=direction.value,
            direction_angle=direction_angles[direction],
            target_found=len(views_with_target) > 0,
            views_with_target=views_with_target,
            prominence={k: round(v, 3) for k, v in prominences.items()},
            best_view=best_view,
            target_area_ratio=max_area / 5,  # Convert back from scaled
            reached=reached,
            obstacles_detected=obstacles_count,
            message=f"Direction: {direction.value} ({direction_angles[direction]}°)" +
                    (f" - Target in: {', '.join(views_with_target)}" if views_with_target else " - Not found")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/single")
async def detect_single(
    image: UploadFile = File(...),
    confidence: float = Form(0.4)
):
    """
    Simple detection endpoint - returns all detected objects.
    No tracking, no target filtering.
    """
    try:
        contents = await image.read()
        img = decode_image(contents)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        detections = run_detection(img, None, confidence)
        
        return {
            "success": True,
            "count": len(detections),
            "detections": detections
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === Stream Processing Endpoints ===

@app.post("/stream/start")
async def start_stream(config: StreamConfig):
    """
    Start processing frames from Pi's stream.
    
    - **stream_url**: Pi stream server URL (e.g., http://192.168.1.105:8080)
    - **target_class**: Object to track
    - **confidence**: Detection threshold
    - **interval_ms**: Processing interval in milliseconds
    """
    try:
        stream_processor.start(config)
        return {
            "success": True,
            "message": f"Stream processing started: {config.stream_url}",
            "config": config.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stream/stop")
async def stop_stream():
    """Stop stream processing."""
    stream_processor.stop()
    return {"success": True, "message": "Stream processing stopped"}


@app.get("/stream/status")
async def stream_status():
    """Get current stream processing status and latest result."""
    result = stream_processor.get_latest()
    return {
        "active": stream_processor.running,
        "stream_url": stream_processor.stream_url,
        "target_class": stream_processor.target_class,
        "latest_result": result
    }


@app.get("/stream/direction")
async def stream_direction():
    """
    Quick endpoint to get just the direction from stream.
    Useful for simple polling from Pi.
    """
    result = stream_processor.get_latest()
    if result:
        return {
            "direction": result["direction"],
            "target_found": result["target_found"],
            "timestamp": result["timestamp"]
        }
    return {"direction": "none", "target_found": False, "timestamp": 0}


@app.get("/detect/realtime/stream", response_model=RealtimeResponse)
async def detect_realtime_stream():
    """
    Real-time detection using video stream from Raspberry Pi.
    Returns the latest result from the active stream processor.
    
    Note: Start the stream first using /stream/start endpoint.
    """
    if not stream_processor.running:
        raise HTTPException(
            status_code=400, 
            detail="Stream processor not running. Start it with /stream/start"
        )
    
    result = stream_processor.get_latest()
    if not result:
        return RealtimeResponse(
            success=True,
            direction="none",
            target_found=False,
            message="Waiting for stream frames..."
        )
    
    # Convert stream result to RealtimeResponse format
    best_target = result.get("target")
    if best_target:
        tracking_info = None
        if result.get("tracking"):
            tracking_info = TrackingInfo(**result["tracking"])
        
        return RealtimeResponse(
            success=True,
            direction=result["direction"],
            target_found=True,
            target_detection=Detection(
                class_name=best_target["class_name"],
                confidence=best_target["confidence"],
                bbox=best_target["bbox"],
                centroid=best_target["centroid"],
                area=best_target["area"],
                area_ratio=best_target["area_ratio"]
            ) if best_target else None,
            tracking=tracking_info,
            all_detections=[Detection(**{k: v for k, v in d.items() 
                                        if k in Detection.__fields__}) 
                           for d in result.get("all_detections", [])],
            distance_ratio=result.get("area_ratio", 0.0),
            reached=result.get("reached", False),
            message=f"Target '{stream_processor.target_class}' found - {result['direction']}" + 
                    (" - REACHED!" if result.get("reached") else "")
        )
    else:
        return RealtimeResponse(
            success=True,
            direction="none",
            target_found=False,
            all_detections=[Detection(**{k: v for k, v in d.items() 
                                        if k in Detection.__fields__}) 
                           for d in result.get("all_detections", [])],
            message=f"Target '{stream_processor.target_class}' not found"
        )


@app.post("/tracker/reset")
async def reset_tracker():
    """Reset the tracking state."""
    global tracker
    tracker = ServerTracker()
    return {"success": True, "message": "Tracker reset"}


# === Manual Control Endpoints ===

# Store manual control commands (simple queue)
manual_control_queue = []
manual_control_lock = threading.Lock()

@app.post("/control/command")
async def send_control_command(command: str = Form(...)):
    """
    Send manual control command to robot.
    
    Commands: stop, forward, backward, left, right, far_left, far_right, 
              rotate_180, servo_left, servo_center, servo_right
    """
    valid_commands = [
        "stop", "forward", "backward", "left", "right", 
        "far_left", "far_right", "rotate_180",
        "servo_left", "servo_center", "servo_right"
    ]
    
    if command not in valid_commands:
        raise HTTPException(status_code=400, detail=f"Invalid command. Valid: {valid_commands}")
    
    with manual_control_lock:
        manual_control_queue.append(command)
    
    return {"success": True, "command": command, "message": f"Command '{command}' queued"}


@app.get("/control/poll")
async def poll_control_command():
    """Poll for manual control commands (used by Pi client)."""
    with manual_control_lock:
        if manual_control_queue:
            cmd = manual_control_queue.pop(0)
            return {"command": cmd, "has_command": True}
        return {"command": None, "has_command": False}


@app.get("/control/clear")
async def clear_control_queue():
    """Clear all queued manual control commands."""
    with manual_control_lock:
        manual_control_queue.clear()
    return {"success": True, "message": "Control queue cleared"}


# === Web UI Endpoint ===

@app.get("/ui", response_class=HTMLResponse)
async def web_ui():
    """Web UI for robot control."""
    return get_web_ui_html()


def get_web_ui_html():
    """Generate web UI HTML."""
    local_ip = get_local_ip()
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Robot Vision Control Panel</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            color: #333;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .ip-display {{
            background: rgba(255,255,255,0.2);
            padding: 15px;
            border-radius: 10px;
            margin-top: 15px;
            font-size: 1.2em;
            font-family: monospace;
        }}
        
        .content {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            padding: 30px;
        }}
        
        .panel {{
            background: #f8f9fa;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .panel h2 {{
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.5em;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        
        .control-group {{
            margin-bottom: 20px;
        }}
        
        .control-group label {{
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #555;
        }}
        
        select, input {{
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 1em;
            transition: border-color 0.3s;
        }}
        
        select:focus, input:focus {{
            outline: none;
            border-color: #667eea;
        }}
        
        .button-group {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 15px;
        }}
        
        button {{
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            flex: 1;
            min-width: 120px;
        }}
        
        button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }}
        
        button:active {{
            transform: translateY(0);
        }}
        
        .btn-primary {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        
        .btn-success {{
            background: #4ade80;
            color: white;
        }}
        
        .btn-danger {{
            background: #ef4444;
            color: white;
        }}
        
        .btn-warning {{
            background: #f59e0b;
            color: white;
        }}
        
        .btn-info {{
            background: #3b82f6;
            color: white;
        }}
        
        .btn-secondary {{
            background: #6b7280;
            color: white;
        }}
        
        .video-container {{
            position: relative;
            background: #000;
            border-radius: 10px;
            overflow: hidden;
            aspect-ratio: 4/3;
            margin-top: 15px;
        }}
        
        .video-container img {{
            width: 100%;
            height: 100%;
            object-fit: contain;
        }}
        
        .video-placeholder {{
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #999;
            font-size: 1.2em;
        }}
        
        .status-display {{
            background: #1a1a2e;
            color: #4ade80;
            padding: 15px;
            border-radius: 10px;
            font-family: monospace;
            font-size: 0.9em;
            margin-top: 15px;
            max-height: 200px;
            overflow-y: auto;
        }}
        
        .status-line {{
            margin: 5px 0;
            padding: 5px;
            border-left: 3px solid #4ade80;
            padding-left: 10px;
        }}
        
        .manual-control {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-top: 15px;
        }}
        
        .manual-control button {{
            padding: 20px;
            font-size: 1.2em;
            min-width: auto;
        }}
        
        .center-btn {{
            grid-column: 2;
        }}
        
        .full-width {{
            grid-column: 1 / -1;
        }}
        
        .status-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 600;
            margin-left: 10px;
        }}
        
        .status-online {{
            background: #4ade80;
            color: white;
        }}
        
        .status-offline {{
            background: #ef4444;
            color: white;
        }}
        
        @media (max-width: 968px) {{
            .content {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Robot Vision Control Panel</h1>
            <div class="ip-display">
                Server IP: <strong id="serverIp">{local_ip if local_ip else 'localhost'}</strong>:8000
            </div>
        </div>
        
        <div class="content">
            <!-- Control Panel -->
            <div class="panel">
                <h2>🎮 Control Panel</h2>
                
                <div class="control-group">
                    <label>Operation Mode</label>
                    <select id="modeSelect">
                        <option value="realtime">Real-time Tracking</option>
                        <option value="multiview">Multi-view Finding</option>
                        <option value="manual">Manual Control</option>
                    </select>
                </div>
                
                <div class="control-group">
                    <label>Target Object to Track</label>
                    <input type="text" id="targetClassInput" placeholder="Enter object name (e.g., person, car, dog)" value="person" list="objectSuggestions">
                    <datalist id="objectSuggestions">
                        <option value="person">
                        <option value="car">
                        <option value="dog">
                        <option value="cat">
                        <option value="bicycle">
                        <option value="motorcycle">
                        <option value="bus">
                        <option value="truck">
                        <option value="bird">
                        <option value="horse">
                    </datalist>
                    <small style="color: #666; font-size: 0.85em; margin-top: 5px; display: block;">
                        Enter any COCO class name. Common: person, car, dog, cat, bicycle, etc.
                    </small>
                </div>
                
                <div class="control-group">
                    <label>Confidence Threshold</label>
                    <input type="range" id="confidence" min="0.1" max="1.0" step="0.1" value="0.4">
                    <span id="confidenceValue">0.4</span>
                </div>
                
                <div class="control-group">
                    <label>Pi Stream URL</label>
                    <input type="text" id="streamUrl" placeholder="http://192.168.1.105:8080" value="">
                </div>
                
                <div class="button-group">
                    <button class="btn-success" onclick="startOperation()">▶ Start</button>
                    <button class="btn-danger" onclick="stopOperation()">⏹ Stop</button>
                    <button class="btn-info" onclick="toggleLiveFeed()">📹 Live Feed</button>
                </div>
                
                <div class="status-display" id="statusDisplay">
                    <div class="status-line">Ready. Select mode and click Start.</div>
                </div>
            </div>
            
            <!-- Video Feed & Manual Control -->
            <div class="panel">
                <h2>📹 Live Feed <span class="status-badge status-offline" id="feedStatus">OFF</span></h2>
                
                <div class="video-container" id="videoContainer">
                    <div class="video-placeholder">Click "Live Feed" to start</div>
                </div>
                
                <!-- Real-time Tracking Display -->
                <div id="trackingDisplay" style="display: none; margin-top: 15px; background: #f8f9fa; padding: 15px; border-radius: 10px; border: 2px solid #667eea;">
                    <h3 style="margin-bottom: 10px; color: #667eea;">🎯 Real-time Tracking</h3>
                    <div id="trackingInfo" style="font-family: monospace; font-size: 0.9em; line-height: 1.8;">
                        <div><strong>Status:</strong> <span id="trackingStatus" style="font-weight: bold;">Waiting...</span></div>
                        <div><strong>Direction:</strong> <span id="trackingDirection">-</span></div>
                        <div><strong>Confidence:</strong> <span id="trackingConfidence">-</span></div>
                        <div><strong>Distance:</strong> <span id="trackingDistance">-</span></div>
                        <div><strong>Velocity:</strong> <span id="trackingVelocity">-</span></div>
                    </div>
                </div>
                
                <div id="manualControlPanel" style="display: none; margin-top: 20px;">
                    <h3 style="margin-bottom: 15px; color: #667eea;">🎮 Manual Control</h3>
                    <div class="manual-control">
                        <button class="btn-secondary" onclick="sendCommand('servo_left')">↖ Servo L</button>
                        <button class="btn-primary" onclick="sendCommand('forward')">↑ Forward</button>
                        <button class="btn-secondary" onclick="sendCommand('servo_right')">↗ Servo R</button>
                        
                        <button class="btn-primary" onclick="sendCommand('left')">← Left</button>
                        <button class="btn-danger" onclick="sendCommand('stop')">⏹ Stop</button>
                        <button class="btn-primary" onclick="sendCommand('right')">→ Right</button>
                        
                        <button class="btn-secondary" onclick="sendCommand('far_left')">↖ Far Left</button>
                        <button class="btn-primary" onclick="sendCommand('backward')">↓ Backward</button>
                        <button class="btn-secondary" onclick="sendCommand('far_right')">↗ Far Right</button>
                        
                        <button class="btn-warning full-width" onclick="sendCommand('rotate_180')">🔄 Rotate 180°</button>
                        <button class="btn-secondary full-width" onclick="sendCommand('servo_center')">🎯 Servo Center</button>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const API_BASE = window.location.origin;
        let liveFeedInterval = null;
        let statusInterval = null;
        let currentMode = 'manual';
        
        // Update confidence display
        document.getElementById('confidence').addEventListener('input', (e) => {{
            document.getElementById('confidenceValue').textContent = e.target.value;
        }});
        
        // Mode change handler
        document.getElementById('modeSelect').addEventListener('change', (e) => {{
            currentMode = e.target.value;
            const manualPanel = document.getElementById('manualControlPanel');
            if (e.target.value === 'manual') {{
                manualPanel.style.display = 'block';
            }} else {{
                manualPanel.style.display = 'none';
            }}
        }});
        
        function addStatus(message, type = 'info') {{
            const statusDisplay = document.getElementById('statusDisplay');
            const line = document.createElement('div');
            line.className = 'status-line';
            line.textContent = `[${{new Date().toLocaleTimeString()}}] ${{message}}`;
            statusDisplay.appendChild(line);
            statusDisplay.scrollTop = statusDisplay.scrollHeight;
        }}
        
        async function startOperation() {{
            const mode = document.getElementById('modeSelect').value;
            const targetClass = document.getElementById('targetClassInput').value.trim().toLowerCase();
            const confidence = parseFloat(document.getElementById('confidence').value);
            const streamUrl = document.getElementById('streamUrl').value;
            
            if (!targetClass) {{
                alert('Please enter an object name to track');
                return;
            }}
            
            addStatus(`Starting ${{mode}} mode for: ${{targetClass}}...`);
            
            try {{
                if (mode === 'realtime' || mode === 'multiview') {{
                    if (!streamUrl) {{
                        alert('Please enter Pi Stream URL');
                        return;
                    }}
                    
                    const response = await fetch(`${{API_BASE}}/stream/start`, {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{
                            stream_url: streamUrl,
                            target_class: targetClass,
                            confidence: confidence,
                            interval_ms: 100
                        }})
                    }});
                    
                    const data = await response.json();
                    if (data.success) {{
                        addStatus(`Stream started: ${{streamUrl}}`);
                        addStatus(`Tracking: ${{targetClass}}`);
                        document.getElementById('trackingDisplay').style.display = 'block';
                        startStatusPolling();
                    }} else {{
                        addStatus(`Error: ${{data.detail || 'Failed to start'}}`, 'error');
                    }}
                }} else {{
                    addStatus('Manual control mode active');
                }}
            }} catch (error) {{
                addStatus(`Error: ${{error.message}}`, 'error');
            }}
        }}
        
        async function stopOperation() {{
            addStatus('Stopping operation...');
            
            try {{
                await fetch(`${{API_BASE}}/stream/stop`, {{ method: 'POST' }});
                stopStatusPolling();
                document.getElementById('trackingDisplay').style.display = 'none';
                addStatus('Operation stopped');
            }} catch (error) {{
                addStatus(`Error: ${{error.message}}`, 'error');
            }}
        }}
        
        // Load available classes on page load
        async function loadAvailableClasses() {{
            try {{
                const response = await fetch(`${{API_BASE}}/classes`);
                const data = await response.json();
                const datalist = document.getElementById('objectSuggestions');
                
                // Clear existing options
                datalist.innerHTML = '';
                
                // Add all classes to datalist
                data.classes.forEach(className => {{
                    const option = document.createElement('option');
                    option.value = className;
                    datalist.appendChild(option);
                }});
                
                addStatus(`Loaded ${{data.classes.length}} available object classes`);
            }} catch (error) {{
                console.error('Failed to load classes:', error);
            }}
        }}
        
        function toggleLiveFeed() {{
            const streamUrl = document.getElementById('streamUrl').value;
            if (!streamUrl) {{
                alert('Please enter Pi Stream URL');
                return;
            }}
            
            if (liveFeedInterval) {{
                clearInterval(liveFeedInterval);
                liveFeedInterval = null;
                document.getElementById('feedStatus').textContent = 'OFF';
                document.getElementById('feedStatus').className = 'status-badge status-offline';
                document.getElementById('videoContainer').innerHTML = '<div class="video-placeholder">Click "Live Feed" to start</div>';
                addStatus('Live feed stopped');
            }} else {{
                // Remove trailing slash from streamUrl if present to avoid double slash
                const cleanUrl = streamUrl.replace(/\/$/, '');
                const feedUrl = `${{cleanUrl}}/video_feed`;
                console.log('Loading live feed from:', feedUrl);
                const img = document.createElement('img');
                img.src = feedUrl;
                img.onerror = () => {{
                    addStatus('Failed to load feed. Check stream URL and ensure Pi stream server is running.', 'error');
                    console.error('Failed to load feed from:', feedUrl);
                    toggleLiveFeed();
                }};
                
                document.getElementById('videoContainer').innerHTML = '';
                document.getElementById('videoContainer').appendChild(img);
                document.getElementById('feedStatus').textContent = 'ON';
                document.getElementById('feedStatus').className = 'status-badge status-online';
                addStatus('Live feed started');
            }}
        }}
        
        async function sendCommand(command) {{
            try {{
                const formData = new FormData();
                formData.append('command', command);
                
                const response = await fetch(`${{API_BASE}}/control/command`, {{
                    method: 'POST',
                    body: formData
                }});
                
                const data = await response.json();
                if (data.success) {{
                    addStatus(`Command sent: ${{command}}`);
                }} else {{
                    addStatus(`Error: ${{data.detail}}`, 'error');
                }}
            }} catch (error) {{
                addStatus(`Error: ${{error.message}}`, 'error');
            }}
        }}
        
        function startStatusPolling() {{
            if (statusInterval) return;
            
            statusInterval = setInterval(async () => {{
                try {{
                    const response = await fetch(`${{API_BASE}}/detect/realtime/stream`);
                    if (response.status === 200) {{
                        const data = await response.json();
                        
                        // Update tracking display
                        const trackingDisplay = document.getElementById('trackingDisplay');
                        if (data.target_found) {{
                            document.getElementById('trackingStatus').textContent = '✅ Tracking';
                            document.getElementById('trackingStatus').style.color = '#4ade80';
                            document.getElementById('trackingDirection').textContent = data.direction.toUpperCase();
                            document.getElementById('trackingConfidence').textContent = (data.target_detection?.confidence * 100).toFixed(1) + '%';
                            document.getElementById('trackingDistance').textContent = (data.distance_ratio * 100).toFixed(1) + '%';
                            
                            if (data.tracking) {{
                                const vel = data.tracking.velocity;
                                document.getElementById('trackingVelocity').textContent = `(${{vel[0].toFixed(1)}}, ${{vel[1].toFixed(1)}}) px/s`;
                            }} else {{
                                document.getElementById('trackingVelocity').textContent = 'Calculating...';
                            }}
                            
                            if (data.reached) {{
                                document.getElementById('trackingStatus').textContent = '🎯 REACHED!';
                                document.getElementById('trackingStatus').style.color = '#f59e0b';
                            }}
                            
                            addStatus(`Target found: ${{data.direction}} (Conf: ${{(data.target_detection?.confidence * 100).toFixed(1)}}%)`);
                        }} else {{
                            document.getElementById('trackingStatus').textContent = '❌ Not Found';
                            document.getElementById('trackingStatus').style.color = '#ef4444';
                            document.getElementById('trackingDirection').textContent = '-';
                            document.getElementById('trackingConfidence').textContent = '-';
                            document.getElementById('trackingDistance').textContent = '-';
                            document.getElementById('trackingVelocity').textContent = '-';
                        }}
                    }}
                }} catch (error) {{
                    // Silent fail
                }}
            }}, 500); // Poll every 500ms for real-time updates
        }}
        
        function stopStatusPolling() {{
            if (statusInterval) {{
                clearInterval(statusInterval);
                statusInterval = null;
            }}
        }}
        
        // Initialize
        addStatus('Control panel ready');
        loadAvailableClasses();
    </script>
</body>
</html>
"""


# === Run Server ===

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
