# Live Stream Setup Guide

This guide explains how to run `client_rpi.py` to get a live camera stream with object detection.

---

## 📋 Prerequisites

1. **Vision Server Running** (on PC/server with GPU)
2. **Raspberry Pi** with USB camera connected
3. **Network Connection** between Pi and server
4. **Dependencies Installed** on both machines

---

## 🚀 Step-by-Step Setup

### Step 1: Start Vision Server (PC/Server)

On your PC or server machine:

```bash
# Navigate to project directory
cd /path/to/Abiogensis_stero

# Start the vision server
python server.py

# Or using uvicorn directly:
uvicorn server:app --host 0.0.0.0 --port 8000
```

**Expected Output:**
```
✓ GPU detected: NVIDIA GeForce RTX 3050
Loading YOLOv11n model on CUDA...
✓ Model loaded! 80 classes available
  Device: CUDA

============================================================
  SERVER STARTING
============================================================
  Access the server at:
    • http://localhost:8000/ (same machine)
    • http://127.0.0.1:8000/ (same machine)
    • http://192.168.1.100:8000/ (from other devices on network)
  API docs: http://localhost:8000/docs
============================================================
```

**Note the server IP address** - you'll need it for the Pi client.

---

### Step 2: Run Client on Raspberry Pi

On your Raspberry Pi:

#### Option A: Automatic Stream Server (Recommended)

The client will automatically start the stream server for you:

```bash
# Basic usage - tracks "person" by default
python client_rpi.py --server http://SERVER_IP:8000 --mode realtime

# Example with actual IP:
python client_rpi.py --server http://192.168.1.100:8000 --mode realtime

# Track a different object
python client_rpi.py --server http://192.168.1.100:8000 --mode realtime --target car

# Use different camera index (if you have multiple cameras)
python client_rpi.py --server http://192.168.1.100:8000 --mode realtime --camera 1

# Adjust confidence threshold
python client_rpi.py --server http://192.168.1.100:8000 --mode realtime --target dog --confidence 0.5

# Use different stream port
python client_rpi.py --server http://192.168.1.100:8000 --mode realtime --stream-port 8081
```

**What happens:**
1. Client initializes camera
2. Client automatically starts `rpi_stream_server.py` in background
3. Stream server runs on port 8080 (or your specified port)
4. Client tells vision server to start processing the stream
5. Detection results are displayed in real-time

**Expected Output:**
```
==================================================
  INITIALIZING ROBOT CONTROLLER
==================================================

[1/3] Initializing camera...
✓ USB Webcam found and working (index 0)
  Resolution: 640x480
✓ Camera test capture successful

[2/3] Initializing motor controller...
✓ Arduino connected on /dev/ttyUSB0

[3/3] Connecting to vision server...
✓ Connected to server: online
  Model loaded: True

✓ Robot controller initialized successfully!
==================================================

==================================================
  REAL-TIME TRACKING MODE (Video Stream)
  Target: person
==================================================

Starting local stream server...
Starting stream server: python rpi_stream_server.py --port 8080 --camera 0 --width 640 --height 480
Waiting for stream server to start at http://192.168.1.105:8080...
  Attempt 1/10...
  Attempt 2/10...
✓ Stream server is running!
  Status: streaming
  Resolution: 640x480
  FPS: 28.5
✓ Stream server ready at http://192.168.1.105:8080

Starting server-side stream processing...
  Stream URL: http://192.168.1.105:8080
  Target: person
  Confidence: 0.4
✓ Server stream processing started

Using stream mode (server pulls frames automatically)

Target: dir=center, area=15.2%, vel=(2.1, -1.5)
Target: dir=left, area=12.8%, vel=(5.3, 0.2)
...
```

#### Option B: Manual Stream Server (Alternative)

If you want to run the stream server separately:

**Terminal 1 - Stream Server:**
```bash
python rpi_stream_server.py --port 8080 --camera 0
```

**Terminal 2 - Client:**
```bash
python client_rpi.py --server http://SERVER_IP:8000 --mode realtime --no-stream
```

---

### Step 3: Access Live Stream

Once the client is running, you can access the live stream in several ways:

#### Method 1: Web Browser (Easiest)

1. **Find Pi IP Address:**
   ```bash
   # On Raspberry Pi, run:
   hostname -I
   # Or:
   ip addr show
   ```

2. **Open in Browser:**
   ```
   http://PI_IP:8080/
   ```
   Example: `http://192.168.1.105:8080/`

3. **You'll see:**
   - Live video feed
   - Snapshot button
   - Fullscreen button

#### Method 2: Web UI Control Panel

1. **Open Vision Server Web UI:**
   ```
   http://SERVER_IP:8000/ui
   ```
   Example: `http://192.168.1.100:8000/ui`

2. **Enter Pi Stream URL:**
   - In the "Pi Stream URL" field, enter: `http://PI_IP:8080`
   - Example: `http://192.168.1.105:8080`

3. **Click "Live Feed"** button

4. **Start Tracking:**
   - Enter object name (e.g., "person", "car", "dog")
   - Select "Real-time Tracking" mode
   - Click "Start"
   - View live tracking information

#### Method 3: Direct Stream URL

**MJPEG Stream (for VLC or other players):**
```
http://PI_IP:8080/video_feed
```

**Single Snapshot:**
```
http://PI_IP:8080/snapshot
```

**API Frame (for programmatic access):**
```
http://PI_IP:8080/api/frame
```

---

## 📝 Command Line Options

### client_rpi.py Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--server` | `http://localhost:8000` | Vision server URL |
| `--port` | `/dev/ttyUSB0` | Arduino serial port |
| `--target` | `person` | Object class to track |
| `--mode` | `realtime` | Mode: `realtime`, `multiview`, or `manual` |
| `--confidence` | `0.4` | Detection confidence (0.0-1.0) |
| `--camera` | `0` | USB camera index |
| `--stream-port` | `8080` | Stream server port |
| `--no-stream` | `False` | Disable auto stream server |

### Examples

```bash
# Track a person with default settings
python client_rpi.py --server http://192.168.1.100:8000

# Track a car with higher confidence
python client_rpi.py --server http://192.168.1.100:8000 --target car --confidence 0.6

# Use camera index 1
python client_rpi.py --server http://192.168.1.100:8000 --camera 1

# Manual control mode (no tracking, just control via web UI)
python client_rpi.py --server http://192.168.1.100:8000 --mode manual

# Multi-view mode (3-view scanning)
python client_rpi.py --server http://192.168.1.100:8000 --mode multiview --target person
```

---

## 🔍 Troubleshooting

### Issue: "Camera not found"

**Solution:**
```bash
# Check available cameras
ls -l /dev/video*

# Try different camera index
python client_rpi.py --server http://SERVER_IP:8000 --camera 1
```

### Issue: "Stream server not responding"

**Solution:**
1. Check if port 8080 is available:
   ```bash
   netstat -tuln | grep 8080
   ```

2. Check firewall:
   ```bash
   sudo ufw allow 8080
   ```

3. Verify stream server manually:
   ```bash
   python rpi_stream_server.py --port 8080
   # Then visit http://PI_IP:8080/ in browser
   ```

### Issue: "Cannot connect to server"

**Solution:**
1. Verify server is running:
   ```bash
   curl http://SERVER_IP:8000/
   ```

2. Check network connectivity:
   ```bash
   ping SERVER_IP
   ```

3. Verify firewall allows port 8000

### Issue: "Object not found"

**Solution:**
1. Make sure stream is started from web UI first
2. Check target class name is correct (lowercase: "person", "car", etc.)
3. Lower confidence threshold:
   ```bash
   python client_rpi.py --server http://SERVER_IP:8000 --confidence 0.3
   ```
4. Verify camera feed is working:
   - Visit `http://PI_IP:8080/` in browser
   - You should see live video

### Issue: "Stream processor not running"

**Solution:**
1. Start stream from web UI:
   - Go to `http://SERVER_IP:8000/ui`
   - Enter Pi Stream URL
   - Click "Start"

2. Or start via API:
   ```bash
   curl -X POST http://SERVER_IP:8000/stream/start \
     -H "Content-Type: application/json" \
     -d '{
       "stream_url": "http://PI_IP:8080",
       "target_class": "person",
       "confidence": 0.4
     }'
   ```

---

## 🎯 Quick Start Commands

### Complete Setup (Copy & Paste)

**On Server (PC):**
```bash
python server.py
```

**On Raspberry Pi:**
```bash
# Get your server IP first (replace with actual IP)
python client_rpi.py --server http://192.168.1.100:8000 --mode realtime --target person
```

**Then access:**
- **Live Stream:** `http://PI_IP:8080/`
- **Web UI:** `http://SERVER_IP:8000/ui`
- **API Docs:** `http://SERVER_IP:8000/docs`

---

## 📊 Monitoring

### Check Stream Status

```bash
# On server
curl http://SERVER_IP:8000/stream/status

# On Pi
curl http://PI_IP:8080/api/status
```

### View Detection Results

```bash
# Get latest detection
curl http://SERVER_IP:8000/detect/realtime/stream

# Get direction only
curl http://SERVER_IP:8000/stream/direction
```

---

## 💡 Tips

1. **Find IP Addresses:**
   - Server IP: Check server terminal output or run `ipconfig` (Windows) / `ifconfig` (Linux)
   - Pi IP: Run `hostname -I` on Pi

2. **Test Camera First:**
   ```bash
   python rpi_stream_server.py --port 8080
   # Then visit http://PI_IP:8080/ in browser
   ```

3. **Use Web UI:**
   - Easiest way to control everything
   - Access at `http://SERVER_IP:8000/ui`
   - Enter object name, start tracking, view live feed

4. **Check Logs:**
   - Client shows detailed status messages
   - Stream server shows FPS and status
   - Server shows detection results

---

## ✅ Success Indicators

You'll know everything is working when you see:

1. **Client Output:**
   ```
   ✓ Stream server is running!
   ✓ Server stream processing started
   Target: dir=center, area=15.2%
   ```

2. **Browser:**
   - Live video feed visible at `http://PI_IP:8080/`
   - Web UI shows tracking information

3. **Detection:**
   - Status messages showing "Target found"
   - Direction updates (left/center/right)
   - Confidence and distance displayed

---

## 🆘 Need Help?

1. Check all services are running
2. Verify network connectivity
3. Check camera permissions: `sudo usermod -a -G video $USER`
4. Review error messages in terminal
5. Test each component separately

