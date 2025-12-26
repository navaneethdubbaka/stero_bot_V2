# API Endpoint Documentation

This document provides detailed explanations for all API endpoints in the Robot Vision System.

---

## Table of Contents

1. [Vision Server (server.py)](#vision-server-serverpy)
2. [Raspberry Pi Stream Server (rpi_stream_server.py)](#raspberry-pi-stream-server-rpi_stream_serverpy)

---

## Vision Server (server.py)

**Base URL:** `http://localhost:8000` (or your server IP)

The Vision Server runs YOLOv11n object detection on GPU and provides various detection endpoints.

---

### Health & Information Endpoints

#### `GET /`

**Description:** Health check endpoint that returns server status and basic information.

**Request:**
- Method: `GET`
- No parameters required

**Response:**
```json
{
  "status": "online",
  "model_loaded": true,
  "available_classes": ["person", "bicycle", "car", ...],
  "stream_active": false
}
```

**Fields:**
- `status`: Server status (always "online" if server is running)
- `model_loaded`: Whether YOLO model is loaded (always true)
- `available_classes`: List of first 30 COCO class names
- `stream_active`: Whether stream processor is currently running

**Usage Example:**
```bash
curl http://localhost:8000/
```

**Use Case:** Check if server is running and ready to process requests.

---

#### `GET /classes`

**Description:** Get all 80 available COCO class names that YOLO can detect.

**Request:**
- Method: `GET`
- No parameters required

**Response:**
```json
{
  "classes": [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    ...
  ]
}
```

**Usage Example:**
```bash
curl http://localhost:8000/classes
```

**Use Case:** Get list of all detectable object classes to use as `target_class` parameter.

---

### Detection Endpoints

#### `POST /detect/realtime`

**Description:** Real-time single image detection and tracking. Processes one image, detects objects, tracks the target object across frames, and returns navigation direction.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image` | File | Yes | - | Image file (JPEG/PNG) to process |
| `target_class` | String | No | "person" | COCO class name to track (e.g., "person", "car", "dog") |
| `confidence` | Float | No | 0.4 | Detection confidence threshold (0.0-1.0) |
| `track` | Boolean | No | true | Enable object tracking across frames |

**Response:**
```json
{
  "success": true,
  "direction": "left" | "center" | "right" | "none",
  "target_found": true,
  "target_detection": {
    "class_name": "person",
    "confidence": 0.85,
    "bbox": [100, 150, 300, 450],
    "centroid": [200, 300],
    "area": 60000,
    "area_ratio": 0.15
  },
  "tracking": {
    "object_id": 1,
    "velocity": [5.2, -3.1],
    "frames_tracked": 15
  },
  "all_detections": [...],
  "distance_ratio": 0.15,
  "reached": false,
  "message": "Target 'person' found - left"
}
```

**Response Fields:**
- `success`: Whether request was processed successfully
- `direction`: Navigation direction based on target position:
  - `"left"`: Target in left 35% of frame
  - `"center"`: Target in center 30% of frame
  - `"right"`: Target in right 35% of frame
  - `"none"`: Target not found
- `target_found`: Whether target object was detected
- `target_detection`: Details about the detected target:
  - `bbox`: Bounding box `[x1, y1, x2, y2]`
  - `centroid`: Center point `[x, y]`
  - `area`: Bounding box area in pixels
  - `area_ratio`: Area relative to frame size (0.0-1.0)
- `tracking`: Object tracking information:
  - `object_id`: Unique ID for tracked object
  - `velocity`: Pixel velocity `[vx, vy]` per second
  - `frames_tracked`: Number of frames object has been tracked
- `all_detections`: List of all detected objects (not just target)
- `distance_ratio`: Same as `target_detection.area_ratio`
- `reached`: Whether target is close enough (area_ratio > 0.25)
- `message`: Human-readable status message

**Usage Example:**
```bash
curl -X POST http://localhost:8000/detect/realtime \
  -F "image=@photo.jpg" \
  -F "target_class=person" \
  -F "confidence=0.5" \
  -F "track=true"
```

**Use Case:** Real-time object tracking for robot navigation. Client sends images continuously, server returns direction commands.

**Notes:**
- Tracking persists across multiple requests (same client session)
- Uses server-side tracker to maintain object IDs
- Best target is selected as largest detected object of target class

---

#### `POST /detect/multiview`

**Description:** Multi-view object finding with 5-direction output. Processes three images (left, center, right views) and calculates navigation direction using weighted prominence analysis.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image_left` | File | Yes | - | Left view image (JPEG/PNG) |
| `image_center` | File | Yes | - | Center view image (JPEG/PNG) |
| `image_right` | File | Yes | - | Right view image (JPEG/PNG) |
| `target_class` | String | No | "person" | COCO class name to find |
| `confidence` | Float | No | 0.4 | Detection confidence threshold |

**Response:**
```json
{
  "success": true,
  "direction": "far_left" | "left" | "center" | "right" | "far_right" | "none",
  "direction_angle": 45,
  "target_found": true,
  "views_with_target": ["left", "center"],
  "prominence": {
    "left": 0.6,
    "center": 0.3,
    "right": 0.1
  },
  "best_view": "left",
  "target_area_ratio": 0.12,
  "reached": false,
  "obstacles_detected": 2,
  "message": "Direction: left (45°) - Target in: left, center"
}
```

**Response Fields:**
- `direction`: One of 5 navigation directions:
  - `"far_left"`: Target strongly in left view
  - `"left"`: Target primarily in left view
  - `"center"`: Target in center view
  - `"right"`: Target primarily in right view
  - `"far_right"`: Target strongly in right view
  - `"none"`: Target not found
- `direction_angle`: Angle in degrees (0=far_left, 45=left, 90=center, 135=right, 180=far_right)
- `views_with_target`: List of views where target was detected
- `prominence`: Normalized prominence score (0-1) for each view
- `best_view`: View with highest target prominence
- `target_area_ratio`: Largest area ratio across all views
- `reached`: Whether target is very close (prominence > 0.5)
- `obstacles_detected`: Count of large non-target objects detected
- `message`: Human-readable direction message

**Usage Example:**
```bash
curl -X POST http://localhost:8000/detect/multiview \
  -F "image_left=@left.jpg" \
  -F "image_center=@center.jpg" \
  -F "image_right=@right.jpg" \
  -F "target_class=person" \
  -F "confidence=0.4"
```

**Use Case:** Multi-view scanning mode. Robot captures three images from different servo positions, server determines best navigation direction.

**Notes:**
- Uses weighted prominence calculation to determine 5-direction output
- Obstacle detection counts large objects (>5% frame area) that aren't target
- No tracking - each request is independent

---

#### `POST /detect/single`

**Description:** Simple detection endpoint that returns all detected objects without filtering or tracking.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image` | File | Yes | - | Image file (JPEG/PNG) |
| `confidence` | Float | No | 0.4 | Detection confidence threshold |

**Response:**
```json
{
  "success": true,
  "count": 3,
  "detections": [
    {
      "class_name": "person",
      "confidence": 0.92,
      "bbox": [100, 150, 300, 450],
      "centroid": [200, 300],
      "area": 60000,
      "area_ratio": 0.15,
      "is_target": true
    },
    {
      "class_name": "car",
      "confidence": 0.78,
      "bbox": [400, 200, 600, 400],
      "centroid": [500, 300],
      "area": 40000,
      "area_ratio": 0.10,
      "is_target": false
    }
  ]
}
```

**Response Fields:**
- `success`: Request processed successfully
- `count`: Number of detected objects
- `detections`: List of all detected objects with full details

**Usage Example:**
```bash
curl -X POST http://localhost:8000/detect/single \
  -F "image=@photo.jpg" \
  -F "confidence=0.5"
```

**Use Case:** General object detection without target filtering. Useful for debugging or general scene analysis.

**Notes:**
- No tracking or target filtering
- Returns all detected objects above confidence threshold
- `is_target` field is always false (no target class specified)

---

### Stream Processing Endpoints

#### `POST /stream/start`

**Description:** Start processing frames from Raspberry Pi's video stream. Server will continuously pull frames from Pi's stream endpoint and process them in background.

**Request:**
- Method: `POST`
- Content-Type: `application/json`

**Request Body:**
```json
{
  "stream_url": "http://192.168.1.105:8080",
  "target_class": "person",
  "confidence": 0.4,
  "interval_ms": 100
}
```

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `stream_url` | String | Yes | - | Pi stream server URL (e.g., `http://192.168.1.105:8080`) |
| `target_class` | String | No | "person" | COCO class to track |
| `confidence` | Float | No | 0.4 | Detection confidence threshold |
| `interval_ms` | Integer | No | 100 | Processing interval in milliseconds |

**Response:**
```json
{
  "success": true,
  "message": "Stream processing started: http://192.168.1.105:8080",
  "config": {
    "stream_url": "http://192.168.1.105:8080",
    "target_class": "person",
    "confidence": 0.4,
    "interval_ms": 100
  }
}
```

**Usage Example:**
```bash
curl -X POST http://localhost:8000/stream/start \
  -H "Content-Type: application/json" \
  -d '{
    "stream_url": "http://192.168.1.105:8080",
    "target_class": "person",
    "confidence": 0.4,
    "interval_ms": 100
  }'
```

**Use Case:** Start continuous video stream processing. Server pulls frames from Pi and processes them automatically.

**Notes:**
- Stream processor runs in background thread
- Fetches frames from `{stream_url}/api/frame` endpoint
- Processes frames at specified interval
- Results available via `/detect/realtime/stream` endpoint

---

#### `POST /stream/stop`

**Description:** Stop the active stream processing.

**Request:**
- Method: `POST`
- No parameters required

**Response:**
```json
{
  "success": true,
  "message": "Stream processing stopped"
}
```

**Usage Example:**
```bash
curl -X POST http://localhost:8000/stream/stop
```

**Use Case:** Stop stream processing when done or switching modes.

---

#### `GET /stream/status`

**Description:** Get current stream processing status and latest detection result.

**Request:**
- Method: `GET`
- No parameters required

**Response:**
```json
{
  "active": true,
  "stream_url": "http://192.168.1.105:8080/api/frame",
  "target_class": "person",
  "latest_result": {
    "timestamp": 1234567890.123,
    "direction": "center",
    "target_found": true,
    "target": {...},
    "all_detections": [...],
    "frame_size": [640, 480],
    "area_ratio": 0.15,
    "reached": false,
    "tracking": {...}
  }
}
```

**Response Fields:**
- `active`: Whether stream processor is running
- `stream_url`: Current stream URL being processed
- `target_class`: Currently tracked class
- `latest_result`: Latest detection result (same format as `/detect/realtime/stream`)

**Usage Example:**
```bash
curl http://localhost:8000/stream/status
```

**Use Case:** Check stream status and get latest detection result without polling.

---

#### `GET /stream/direction`

**Description:** Quick endpoint to get just the direction from active stream. Lightweight endpoint for simple polling.

**Request:**
- Method: `GET`
- No parameters required

**Response:**
```json
{
  "direction": "left" | "center" | "right" | "none",
  "target_found": true,
  "timestamp": 1234567890.123
}
```

**Response Fields:**
- `direction`: Current navigation direction
- `target_found`: Whether target was found in latest frame
- `timestamp`: Unix timestamp of latest result

**Usage Example:**
```bash
curl http://localhost:8000/stream/direction
```

**Use Case:** Simple polling for direction commands. Minimal data transfer.

**Notes:**
- Returns `{"direction": "none", "target_found": false, "timestamp": 0}` if stream not active

---

#### `GET /detect/realtime/stream`

**Description:** Get real-time detection result from active video stream. Returns full detection details in same format as `/detect/realtime` but uses stream instead of uploaded image.

**Request:**
- Method: `GET`
- No parameters required

**Response:**
Same format as `POST /detect/realtime` (see above)

**Usage Example:**
```bash
curl http://localhost:8000/detect/realtime/stream
```

**Use Case:** Get full detection results from video stream. Used by client for real-time tracking mode.

**Notes:**
- Requires stream to be started first via `/stream/start`
- Returns error if stream not active
- Returns latest processed frame result
- Includes full tracking information

---

### Tracker Management

#### `POST /tracker/reset`

**Description:** Reset the server-side object tracker. Clears all tracked object IDs and history.

**Request:**
- Method: `POST`
- No parameters required

**Response:**
```json
{
  "success": true,
  "message": "Tracker reset"
}
```

**Usage Example:**
```bash
curl -X POST http://localhost:8000/tracker/reset
```

**Use Case:** Reset tracking state when starting new tracking session or switching targets.

**Notes:**
- Clears all tracked object IDs
- Resets tracking history
- Next detection will start with new object IDs

---

## Raspberry Pi Stream Server (rpi_stream_server.py)

**Base URL:** `http://PI_IP:8080` (default port 8080)

The Raspberry Pi Stream Server captures video from USB webcam and streams it over HTTP.

---

### Web Interface

#### `GET /`

**Description:** Web page with embedded video player for viewing the camera stream.

**Request:**
- Method: `GET`
- No parameters required

**Response:**
- Content-Type: `text/html`
- Returns HTML page with:
  - Live video stream display
  - Snapshot button
  - Fullscreen button
  - Endpoint documentation

**Usage Example:**
```
Open in browser: http://192.168.1.105:8080/
```

**Use Case:** View camera stream in web browser. Anyone on network can access.

**Features:**
- Real-time MJPEG video stream
- Snapshot capture
- Fullscreen mode
- Responsive design

---

### Video Stream Endpoints

#### `GET /video_feed`

**Description:** MJPEG video stream endpoint. Provides continuous video stream in MJPEG format.

**Request:**
- Method: `GET`
- No parameters required

**Response:**
- Content-Type: `multipart/x-mixed-replace; boundary=frame`
- Continuous stream of JPEG frames

**Usage Example:**
```bash
# View in browser
http://192.168.1.105:8080/video_feed

# View in VLC
vlc http://192.168.1.105:8080/video_feed
```

**Use Case:** 
- View stream in web browsers
- Integrate with video players (VLC, ffplay)
- Embed in web pages

**Notes:**
- Stream runs at ~30 FPS
- Each frame is JPEG encoded
- Stream continues until connection closed

---

#### `GET /snapshot`

**Description:** Get single JPEG snapshot from current camera frame.

**Request:**
- Method: `GET`
- No parameters required

**Response:**
- Content-Type: `image/jpeg`
- Single JPEG image

**Usage Example:**
```bash
# Download snapshot
curl http://192.168.1.105:8080/snapshot -o snapshot.jpg

# View in browser
http://192.168.1.105:8080/snapshot
```

**Use Case:** 
- Capture single frame
- Debug camera feed
- Save still images

**Notes:**
- Returns latest captured frame
- Returns 503 if camera not ready

---

### API Endpoints

#### `GET /api/frame`

**Description:** Get raw JPEG frame for API consumption. Used by vision server to pull frames for processing.

**Request:**
- Method: `GET`
- No parameters required

**Response:**
- Content-Type: `image/jpeg`
- Raw JPEG bytes

**Usage Example:**
```bash
curl http://192.168.1.105:8080/api/frame -o frame.jpg
```

**Use Case:** 
- Vision server pulls frames for processing
- Programmatic frame access
- Integration with detection systems

**Notes:**
- Returns latest frame buffer
- Used by `/stream/start` endpoint on vision server
- Returns 503 if camera not ready

---

#### `GET /api/status`

**Description:** Get camera and stream server status information.

**Request:**
- Method: `GET`
- No parameters required

**Response:**
```json
{
  "status": "streaming",
  "resolution": "640x480",
  "fps_target": 30,
  "fps_actual": 28.5,
  "frame_count": 12345,
  "quality": 85
}
```

**Response Fields:**
- `status`: Stream status ("streaming" or "stopped")
- `resolution`: Frame resolution (width x height)
- `fps_target`: Target frames per second
- `fps_actual`: Actual measured FPS
- `frame_count`: Total frames captured since start
- `quality`: JPEG quality (1-100)

**Usage Example:**
```bash
curl http://192.168.1.105:8080/api/status
```

**Use Case:** 
- Monitor stream performance
- Check camera status
- Debug streaming issues

**Notes:**
- FPS calculated over 1-second windows
- Frame count increments continuously

---

## Usage Workflows

### Real-time Tracking with Stream

1. **Start Pi Stream Server:**
   ```bash
   python rpi_stream_server.py --port 8080
   ```

2. **Start Vision Server Stream Processing:**
   ```bash
   curl -X POST http://localhost:8000/stream/start \
     -H "Content-Type: application/json" \
     -d '{"stream_url": "http://PI_IP:8080", "target_class": "person"}'
   ```

3. **Get Detection Results:**
   ```bash
   curl http://localhost:8000/detect/realtime/stream
   ```

### Image-based Detection

1. **Send single image:**
   ```bash
   curl -X POST http://localhost:8000/detect/realtime \
     -F "image=@photo.jpg" \
     -F "target_class=person"
   ```

2. **Multi-view detection:**
   ```bash
   curl -X POST http://localhost:8000/detect/multiview \
     -F "image_left=@left.jpg" \
     -F "image_center=@center.jpg" \
     -F "image_right=@right.jpg"
   ```

---

## Error Responses

All endpoints may return standard HTTP error codes:

- **400 Bad Request:** Invalid parameters or request format
- **500 Internal Server Error:** Server-side processing error
- **503 Service Unavailable:** Service not ready (e.g., camera not initialized)

Error response format:
```json
{
  "detail": "Error message description"
}
```

---

## Notes

- All image endpoints accept JPEG and PNG formats
- Confidence threshold: 0.0 (low) to 1.0 (high)
- Stream processing runs in background thread
- Tracking persists across requests until reset
- GPU acceleration automatically used if available

