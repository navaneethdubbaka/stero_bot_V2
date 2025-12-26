# Robot Vision System - FastAPI Server + Raspberry Pi Client

A client-server robot vision system using **YOLOv11n** for object detection and tracking. The server processes images and returns navigation commands; the Raspberry Pi client captures images and controls motors.

**NEW: Pi Stream Server** - Stream video from Pi that anyone on the network can view!

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              NETWORK                                        │
│                                                                             │
│  ┌─────────────────────┐              ┌─────────────────────────────────┐  │
│  │   RASPBERRY PI      │   HTTP/REST  │         SERVER (PC/Cloud)       │  │
│  │   (client_rpi.py)   │◄────────────►│         (server.py)             │  │
│  │                     │              │                                 │  │
│  │  ┌───────────────┐  │   Images     │  ┌───────────────────────────┐  │  │
│  │  │    Camera     │──┼─────────────►│  │      YOLOv11n Model      │  │  │
│  │  └───────────────┘  │              │  └───────────────────────────┘  │  │
│  │                     │              │               │                 │  │
│  │  ┌───────────────┐  │   Commands   │               ▼                 │  │
│  │  │    Motors     │◄─┼─────────────┤│  ┌───────────────────────────┐  │  │
│  │  │   (Arduino)   │  │   (JSON)    ││  │  Detection + Tracking    │  │  │
│  │  └───────────────┘  │              │  └───────────────────────────┘  │  │
│  │                     │              │               │                 │  │
│  │  ┌───────────────┐  │              │               ▼                 │  │
│  │  │    Servo      │  │              │  ┌───────────────────────────┐  │  │
│  │  │  (Camera Pan) │  │              │  │   Direction Calculation   │  │  │
│  │  └───────────────┘  │              │  └───────────────────────────┘  │  │
│  └─────────────────────┘              └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Files

| File | Location | Description |
|------|----------|-------------|
| `server.py` | Server (PC/Cloud) | FastAPI server with YOLO processing |
| `client_rpi.py` | Raspberry Pi | Image capture + motor control client |
| `rpi_stream_server.py` | Raspberry Pi | **Video streaming server** (view from any browser!) |
| `requirements.txt` | Both | Python dependencies |

---

## 🚀 Quick Start

### 1. Server Setup (PC with GPU recommended)

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python server.py
# or with uvicorn for production:
uvicorn server:app --host 0.0.0.0 --port 8000
```

Server will be available at `http://YOUR_IP:8000`

### 2. Raspberry Pi Setup

```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) For Pi Camera support
pip install picamera2
```

**Note:** USB webcam is the default camera. Use `--picamera` flag to use Pi camera.

#### Option A: Run Stream Server (anyone can view video)

```bash
# USB Webcam (default)
python rpi_stream_server.py --port 8080

# If you have multiple USB cameras, specify index
python rpi_stream_server.py --port 8080 --camera 1

# Pi Camera (if needed)
python rpi_stream_server.py --port 8080 --picamera
```

Now anyone can view the stream at: `http://PI_IP:8080/`

#### Option B: Run Client (sends images to vision server)

```bash
# USB Webcam (default)
python client_rpi.py --server http://SERVER_IP:8000 --mode realtime --target person

# Pi Camera (if needed)
python client_rpi.py --server http://SERVER_IP:8000 --mode realtime --picamera
```

---

## 🔌 API Endpoints

### Vision Server (PC - port 8000)

#### Health Check
```http
GET /
```
Returns server status and available classes.

#### Get All Classes
```http
GET /classes
```
Returns all 80 COCO class names.

---

### Pi Stream Server (Raspberry Pi - port 8080)

| Endpoint | Description |
|----------|-------------|
| `GET /` | Web page with video player |
| `GET /video_feed` | MJPEG stream (for browsers/VLC) |
| `GET /snapshot` | Single JPEG image |
| `GET /api/frame` | Raw JPEG for API consumption |
| `GET /api/status` | Camera status and FPS |

**View in browser:** `http://PI_IP:8080/`

**View in VLC:** `vlc http://PI_IP:8080/video_feed`

---

### Real-time Detection
```http
POST /detect/realtime
```

Single image detection with tracking.

**Request:**
| Field | Type | Description |
|-------|------|-------------|
| `image` | File | JPEG/PNG image |
| `target_class` | string | Object to track (default: "person") |
| `confidence` | float | Threshold 0-1 (default: 0.4) |
| `track` | bool | Enable tracking (default: true) |

**Response:**
```json
{
  "success": true,
  "direction": "left",
  "target_found": true,
  "target_detection": {
    "class_name": "person",
    "confidence": 0.92,
    "bbox": [100, 150, 300, 400],
    "centroid": [200, 275],
    "area": 50000,
    "area_ratio": 0.163
  },
  "tracking": {
    "object_id": 0,
    "velocity": [25.5, -10.2],
    "frames_tracked": 15
  },
  "distance_ratio": 0.163,
  "reached": false,
  "message": "Target 'person' found - left"
}
```

---

### Multi-view Detection
```http
POST /detect/multiview
```

3-image scan with 5-direction output.

**Request:**
| Field | Type | Description |
|-------|------|-------------|
| `image_left` | File | Left view image |
| `image_center` | File | Center view image |
| `image_right` | File | Right view image |
| `target_class` | string | Object to find (default: "person") |
| `confidence` | float | Threshold 0-1 (default: 0.4) |

**Response:**
```json
{
  "success": true,
  "direction": "right",
  "direction_angle": 135,
  "target_found": true,
  "views_with_target": ["center", "right"],
  "prominence": {
    "left": 0.0,
    "center": 0.35,
    "right": 0.65
  },
  "best_view": "right",
  "target_area_ratio": 0.13,
  "reached": false,
  "obstacles_detected": 2,
  "message": "Direction: right (135°) - Target in: center, right"
}
```

---

### Simple Detection
```http
POST /detect/single
```

Basic detection - returns all objects, no tracking.

**Response:**
```json
{
  "success": true,
  "count": 5,
  "detections": [
    {"class_name": "person", "confidence": 0.92, "bbox": [...], ...},
    {"class_name": "chair", "confidence": 0.85, "bbox": [...], ...}
  ]
}
```

---

### Reset Tracker
```http
POST /tracker/reset
```

Clears tracking history (useful when target changes).

---

### Stream Processing (Server pulls from Pi)

The vision server can pull frames directly from Pi's stream:

#### Start Stream Processing
```http
POST /stream/start
Content-Type: application/json

{
  "stream_url": "http://192.168.1.105:8080",
  "target_class": "person",
  "confidence": 0.4,
  "interval_ms": 100
}
```

#### Get Stream Status
```http
GET /stream/status
```

#### Get Latest Direction (for polling)
```http
GET /stream/direction
```

#### Stop Stream
```http
POST /stream/stop
```

---

## 🎯 Detection Modes

### Mode 1: Real-time Tracking (Client Push)

Continuous single-image detection with object tracking. Pi captures and sends images.

```bash
python client_rpi.py --mode realtime --target person
```

**Features:**
- Centroid-based tracking across frames
- Velocity calculation for prediction
- 3-zone direction (left/center/right)
- Distance estimation via area ratio

**Flow:**
```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│ Capture │ ──► │  Send   │ ──► │ Process │ ──► │  Move   │
│  Frame  │     │ to API  │     │ (YOLO)  │     │ Motors  │
└─────────┘     └─────────┘     └─────────┘     └─────────┘
     ▲                                               │
     └───────────────────────────────────────────────┘
                        ~30 FPS loop
```

---

### Mode 2: Multi-view Object Finding (Client Push)

3-view scan with 5-direction navigation.

```bash
python client_rpi.py --mode multiview --target person
```

---

### Mode 3: Stream Processing (Server Pull)

Server pulls frames from Pi's stream server. Good for centralized control.

**On Raspberry Pi:**
```bash
python rpi_stream_server.py --port 8080
```

**On PC (start processing via API):**
```bash
curl -X POST http://localhost:8000/stream/start \
  -H "Content-Type: application/json" \
  -d '{"stream_url": "http://PI_IP:8080", "target_class": "person"}'
```

**Poll for direction:**
```bash
curl http://localhost:8000/stream/direction
# Returns: {"direction": "left", "target_found": true, "timestamp": 1703520000}
```

**5 Directions:**
```
          180° arc
╔═════════╦═════════╦═════════╦═════════╦═════════╗
║FAR_LEFT ║  LEFT   ║ CENTER  ║  RIGHT  ║FAR_RIGHT║
║   0°    ║   45°   ║   90°   ║  135°   ║  180°   ║
╚═════════╩═════════╩═════════╩═════════╩═════════╝
```

**Flow:**
```
┌───────────────┐
│ Rotate Servo  │
│ Left→Center→Right
└───────┬───────┘
        ▼
┌───────────────┐
│ Capture 3     │
│ Images        │
└───────┬───────┘
        ▼
┌───────────────┐
│ Send to API   │
│ /detect/multiview
└───────┬───────┘
        ▼
┌───────────────┐
│ Calculate     │
│ 5-Direction   │
└───────┬───────┘
        ▼
┌───────────────┐
│ Move Robot    │
│ (ML/MC/MR/...)│
└───────────────┘
```

---

## 📐 Direction Calculation

### Real-time (3 zones)

```
┌─────────────────────────────────────────────────────────┐
│             │                             │             │
│    LEFT     │          CENTER             │    RIGHT    │
│   0-35%     │         35%-65%             │  65%-100%   │
│             │                             │             │
└─────────────────────────────────────────────────────────┘
```

### Multi-view (5 directions)

Based on **weighted prominence** across views:

```python
weighted = (-1 × left) + (0 × center) + (1 × right)

# Mapping:
#  -1.0 to -0.6  →  FAR_LEFT   (0°)
#  -0.6 to -0.2  →  LEFT       (45°)
#  -0.2 to  0.2  →  CENTER     (90°)
#   0.2 to  0.6  →  RIGHT      (135°)
#   0.6 to  1.0  →  FAR_RIGHT  (180°)
```

---

## 🔧 Configuration

### Client Command Line

```bash
python client_rpi.py \
  --server http://192.168.1.100:8000 \
  --port /dev/ttyUSB0 \
  --target person \
  --mode realtime \
  --confidence 0.4 \
  --picamera  # Use Pi camera instead of USB webcam
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--server` | localhost:8000 | Server URL |
| `--port` | /dev/ttyUSB0 | Arduino serial port |
| `--target` | person | Object class to track |
| `--mode` | realtime | `realtime` or `multiview` |
| `--confidence` | 0.4 | Detection threshold |
| `--picamera` | false | Use Pi camera |

---

## 📡 Arduino Commands

The client sends these commands to Arduino via serial:

### Movement
| Command | Action |
|---------|--------|
| `MFL` | Move Far Left (sharp turn) |
| `ML` | Move Left |
| `MC` | Move Center (straight) |
| `MR` | Move Right |
| `MFR` | Move Far Right (sharp turn) |
| `ST` | Stop |
| `RB` | Rotate 180° |

### Servo
| Command | Action |
|---------|--------|
| `SL` | Servo Left |
| `SC` | Servo Center |
| `SR` | Servo Right |
| `PHOTO_ACK` | Photo captured ack |

### Arduino → Pi Messages
| Message | Meaning |
|---------|---------|
| `ROTATION_ACK` | Servo rotation done |
| `OBSTACLE DETECTED` | Ultrasonic triggered |
| `OBSTACLE CLEARED` | Path clear |

---

## 🧪 Testing the API

### Using curl

```bash
# Health check
curl http://localhost:8000/

# Real-time detection
curl -X POST http://localhost:8000/detect/realtime \
  -F "image=@test.jpg" \
  -F "target_class=person" \
  -F "confidence=0.4"

# Multi-view detection
curl -X POST http://localhost:8000/detect/multiview \
  -F "image_left=@left.jpg" \
  -F "image_center=@center.jpg" \
  -F "image_right=@right.jpg" \
  -F "target_class=person"
```

### Using Python

```python
import requests

# Real-time
with open("frame.jpg", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/detect/realtime",
        files={"image": f},
        data={"target_class": "person"}
    )
print(resp.json())

# Multi-view
files = {
    "image_left": open("left.jpg", "rb"),
    "image_center": open("center.jpg", "rb"),
    "image_right": open("right.jpg", "rb"),
}
resp = requests.post(
    "http://localhost:8000/detect/multiview",
    files=files,
    data={"target_class": "person"}
)
print(resp.json())
```

---

## 📊 API Documentation

FastAPI provides automatic interactive docs:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 🏷️ COCO Classes (80 objects)

| Category | Classes |
|----------|---------|
| **People** | person |
| **Vehicles** | bicycle, car, motorcycle, airplane, bus, train, truck, boat |
| **Animals** | bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe |
| **Accessories** | backpack, umbrella, handbag, tie, suitcase |
| **Sports** | frisbee, skis, snowboard, sports ball, kite, baseball bat, tennis racket |
| **Kitchen** | bottle, wine glass, cup, fork, knife, spoon, bowl |
| **Food** | banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake |
| **Furniture** | chair, couch, potted plant, bed, dining table, toilet |
| **Electronics** | tv, laptop, mouse, remote, keyboard, cell phone |
| **Appliances** | microwave, oven, toaster, sink, refrigerator |
| **Indoor** | book, clock, vase, scissors, teddy bear, hair drier, toothbrush |

⚠️ **Note**: `door` is NOT in COCO. Train a custom model if needed.

---

## 📈 Performance

| Metric | Server (GPU) | Server (CPU) |
|--------|--------------|--------------|
| Detection latency | ~30ms | ~150ms |
| Requests/sec | ~30 | ~6 |
| Model memory | ~500MB | ~500MB |

---

## 🔮 Future Improvements

- [ ] WebSocket for real-time streaming
- [ ] Multiple client support with session IDs
- [ ] Redis for distributed tracking state
- [ ] Custom model training for doors
- [ ] Docker deployment

---

## 📄 License

MIT License
