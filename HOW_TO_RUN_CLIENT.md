# How to Run client_rpi.py

## Quick Start

### Prerequisites

1. **Install dependencies** on Raspberry Pi:
   ```bash
   pip install -r requirements_rpi.txt
   ```

2. **Ensure the vision server is running** on your PC/server:
   ```bash
   # On your PC/server
   python server.py
   ```
   The server should be accessible at `http://SERVER_IP:8000`

3. **Connect hardware**:
   - USB camera connected to Raspberry Pi
   - Arduino connected via USB serial (if using motors)

---

## Basic Usage

### 1. Real-time Tracking Mode (Default - with Live Stream)

This mode provides **live video feed with object detection visualization** (bounding boxes, labels, tracking info).

```bash
python client_rpi.py --server http://YOUR_SERVER_IP:8000
```

**Example:**
```bash
python client_rpi.py --server http://192.168.1.50:8000
```

**What happens:**
- Camera initializes and starts capturing frames
- Integrated stream server starts automatically on port 8080
- Detection results are fetched from the vision server
- Live video stream with detection overlay is available at:
  - `http://YOUR_PI_IP:8080/` - Web page viewer
  - `http://YOUR_PI_IP:8080/video_feed` - Direct MJPEG stream
- Robot moves automatically based on detection results

**View the live stream:**
- Open browser: `http://YOUR_PI_IP:8080/`
- Or use VLC/other player: `http://YOUR_PI_IP:8080/video_feed`

---

### 2. Real-time Mode with Custom Target Object

```bash
python client_rpi.py --server http://YOUR_SERVER_IP:8000 --target dog
```

**Available target objects:** person, car, dog, cat, bicycle, motorcycle, bus, truck, etc.
(See COCO classes for full list)

---

### 3. Real-time Mode with Custom Settings

```bash
python client_rpi.py \
  --server http://192.168.1.50:8000 \
  --target person \
  --confidence 0.5 \
  --camera 0 \
  --stream-port 8080
```

**Parameters:**
- `--server`: Vision server URL (required)
- `--target`: Object class to track (default: "person")
- `--confidence`: Detection confidence threshold 0.0-1.0 (default: 0.4)
- `--camera`: USB camera index (default: 0, try 1 or 2 if camera not found)
- `--stream-port`: Port for live stream server (default: 8080)

---

### 4. Multi-view Finding Mode

Scans left, center, right views to find objects:

```bash
python client_rpi.py --server http://YOUR_SERVER_IP:8000 --mode multiview --target person
```

---

### 5. Manual Control Mode

Waits for commands from web UI:

```bash
python client_rpi.py --server http://YOUR_SERVER_IP:8000 --mode manual
```

Then use the web UI at `http://YOUR_SERVER_IP:8000/ui` to send control commands.

---

### 6. Image-based Mode (No Stream)

If you want to disable the stream server and use image-based detection:

```bash
python client_rpi.py --server http://YOUR_SERVER_IP:8000 --no-stream
```

---

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--server` | `http://localhost:8000` | Vision server URL |
| `--port` | `/dev/ttyUSB0` | Arduino serial port |
| `--target` | `person` | Target object class to track |
| `--mode` | `realtime` | Mode: `realtime`, `multiview`, or `manual` |
| `--confidence` | `0.4` | Detection confidence threshold (0.0-1.0) |
| `--camera` | `0` | USB camera index (0, 1, 2...) |
| `--stream-port` | `8080` | Port for integrated stream server |
| `--no-stream` | `False` | Disable stream server, use image mode |

---

## Step-by-Step Example

### Complete Setup and Run

**1. On your PC/Server (where server.py runs):**
```bash
# Start the vision server
python server.py
```
Note the server IP address (e.g., `192.168.1.50`)

**2. On Raspberry Pi:**
```bash
# Navigate to project directory
cd /path/to/Abiogensis_stero

# Install dependencies (if not already done)
pip install -r requirements_rpi.txt

# Run the client
python client_rpi.py --server http://192.168.1.50:8000 --target person
```

**3. Output you'll see:**
```
==================================================
  INITIALIZING ROBOT CONTROLLER
==================================================

[1/3] Initializing camera...
Found video devices: /dev/video0
✓ USB Webcam found and working (index 0)
  Resolution: 640x480
✓ Camera test successful - Frame shape: (480, 640, 3)

[2/3] Initializing motor controller...
✓ Arduino connected on /dev/ttyUSB0

[3/3] Connecting to vision server...
✓ Connected to server: Vision Server Running
  Model loaded: True

✓ Robot controller initialized successfully!
==================================================

==================================================
  REAL-TIME TRACKING MODE (Video Stream)
  Target: person
==================================================

Starting integrated stream server with detection visualization...
✓ Capture thread started
Waiting for stream server at http://192.168.1.100:8080...
  Attempt 1/10...
✓ Stream server is running!
  Status: streaming
  Resolution: 640x480

🌐 Stream server running at:
   http://192.168.1.100:8080/
   http://192.168.1.100:8080/video_feed

Starting server-side stream processing...
  Stream URL: http://192.168.1.100:8080
  Target: person
  Confidence: 0.4
✓ Server stream processing started
```

**4. View the live stream:**
- Open browser: `http://192.168.1.100:8080/`
- You'll see live video with:
  - Green bounding boxes around detected target objects
  - Blue boxes for other detected objects
  - Labels with class name and confidence
  - Direction arrows
  - Tracking information
  - Status overlay

---

## Troubleshooting

### Camera Not Found

```bash
# Try different camera index
python client_rpi.py --server http://SERVER_IP:8000 --camera 1

# Check available cameras
lsusb | grep -i camera
ls -l /dev/video*
```

### Cannot Connect to Server

```bash
# Check server is running
curl http://SERVER_IP:8000/

# Check network connectivity
ping SERVER_IP

# Try with explicit IP
python client_rpi.py --server http://192.168.1.50:8000
```

### Stream Server Not Accessible

```bash
# Check firewall
sudo ufw allow 8080

# Check if port is in use
sudo netstat -tulpn | grep 8080

# Try different port
python client_rpi.py --server http://SERVER_IP:8000 --stream-port 8081
```

### Arduino/Serial Issues

```bash
# Check serial port
ls -l /dev/ttyUSB*
ls -l /dev/ttyACM*

# Try different port
python client_rpi.py --server http://SERVER_IP:8000 --port /dev/ttyACM0
```

---

## What You'll See in the Live Stream

The integrated stream server displays:

1. **Bounding Boxes:**
   - 🟢 **Green** = Target object (the one you're tracking)
   - 🔵 **Blue** = Other detected objects
   - 🟡 **Yellow border** = Highlighted target detection

2. **Labels:**
   - Class name (e.g., "person", "dog")
   - Confidence score (e.g., "0.85")

3. **Direction Indicators:**
   - ← Arrow = Object on left
   - → Arrow = Object on right
   - ● Circle = Object centered

4. **Tracking Info:**
   - Object ID
   - Velocity (x, y)

5. **Status Overlay:**
   - Target class name
   - Detection status ("Found: LEFT" or "Searching...")
   - Timestamp
   - FPS counter

---

## Stopping the Client

Press `Ctrl+C` to stop. The client will:
- Stop the stream server
- Stop camera capture
- Stop motors
- Clean up resources

---

## Advanced Usage

### Custom Configuration

You can modify the default values in the `Config` class in `client_rpi.py`:

```python
@dataclass
class Config:
    server_url: str = "http://localhost:8000"
    camera_width: int = 640
    camera_height: int = 480
    target_class: str = "person"
    confidence: float = 0.4
    # ... etc
```

### Running in Background

```bash
# Run in background
nohup python client_rpi.py --server http://SERVER_IP:8000 > client.log 2>&1 &

# Check status
tail -f client.log

# Stop
pkill -f client_rpi.py
```

---

## Next Steps

1. **Start the vision server** on your PC/server
2. **Run client_rpi.py** on Raspberry Pi with appropriate arguments
3. **Open the stream URL** in your browser to see live detection
4. **Use the web UI** at `http://SERVER_IP:8000/ui` for manual control

For more details, see:
- `LIVE_STREAM_GUIDE.md` - Complete streaming guide
- `API_DOCUMENTATION.md` - API reference

