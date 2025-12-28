# How Object Tracking Works

## Overview

The robot uses a **centroid-based tracking algorithm** combined with **YOLO object detection** to continuously track and follow objects in real-time. Here's how it works:

---

## 🔄 Complete Tracking Flow

```
Raspberry Pi Camera → Server (YOLO Detection) → Tracker (Match & Track) → Client (Move Robot)
```

---

## 📊 Step-by-Step Process

### **1. Frame Capture (Raspberry Pi)**
- Camera captures frames at ~30 FPS
- Frames are sent to the server via HTTP stream (`/video_feed`)
- Each frame is a JPEG image

### **2. Object Detection (Server - YOLO)**
- **YOLOv11n model** processes each frame on GPU
- Detects all objects in the frame (person, car, dog, etc.)
- For each detection, extracts:
  - **Bounding box** `[x1, y1, x2, y2]`
  - **Class name** (e.g., "person")
  - **Confidence score** (0.0 to 1.0)
  - **Centroid** `[(x1+x2)/2, (y1+y2)/2]` - center point
  - **Area** (width × height)

### **3. Target Filtering**
- Filters detections to find the **target class** (e.g., "person")
- Only objects matching the target class are considered for tracking

### **4. Object Tracking (ServerTracker)**

The `ServerTracker` class maintains tracking state across frames:

#### **A. Track Matching (Centroid Distance)**
```python
# For each existing track:
1. Get last known centroid position
2. Calculate distance to each new detection's centroid
3. Match to closest detection within max_distance (100 pixels)
4. If match found: Update track with new position
5. If no match: Mark track as lost
```

#### **B. Track Creation**
- New detections that don't match existing tracks get **new object IDs**
- Each track stores a **history** of positions (last 30 frames)

#### **C. Track Cleanup**
- Tracks with no update for **1 second** are removed
- All tracks cleared if no updates for **5 seconds**

### **5. Best Target Selection**
- If multiple targets found, selects the **largest one** (by area)
- This ensures the robot follows the most prominent/close object

### **6. Direction Calculation**
Based on the target's centroid position in the frame:

```
Frame Width Zones:
┌─────────┬─────────┬─────────┐
│  LEFT   │ CENTER  │  RIGHT  │
│  <35%   │ 35-65%  │  >65%   │
└─────────┴─────────┴─────────┘

If centroid X < 35% of width → "left"
If centroid X > 65% of width → "right"
Otherwise → "center"
```

### **7. Velocity Calculation**
Tracks object movement over time:

```python
velocity_x = (current_x - previous_x) / time_delta
velocity_y = (current_y - previous_y) / time_delta
```

- **Positive velocity_x**: Object moving right
- **Negative velocity_x**: Object moving left
- Used to predict object movement and avoid unnecessary robot movement

### **8. Distance Estimation**
- Calculates **area ratio**: `target_area / frame_area`
- If area ratio > **25%**: Target is "reached" (close enough)
- Robot stops when target is reached

### **9. Robot Movement (Client)**
The client receives tracking data and moves the robot:

```python
if target_found:
    if direction == "left":
        → Move robot left
    elif direction == "right":
        → Move robot right
    else:  # center
        → Move robot forward
    
    # Smart prediction: If object moving toward center, wait
    if velocity[0] > 50:  # Moving right
        → Wait (object coming to center)
```

---

## 🎯 Key Tracking Features

### **1. Persistent Object IDs**
- Each tracked object gets a unique `object_id`
- Same object maintains same ID across frames
- Allows tracking multiple objects simultaneously

### **2. Position History**
- Stores last 30 positions for each track
- Enables velocity calculation and smooth tracking

### **3. Distance-Based Matching**
- Uses **Euclidean distance** between centroids
- Maximum matching distance: **100 pixels**
- Prevents false matches when objects are far apart

### **4. Velocity Prediction**
- Calculates object movement speed (pixels/second)
- Helps robot anticipate object movement
- Reduces unnecessary robot movement

### **5. Automatic Track Management**
- Creates new tracks for new objects
- Removes lost tracks automatically
- Handles occlusions and temporary disappearances

---

## 📈 Tracking Data Structure

### **Server Response:**
```json
{
  "success": true,
  "target_found": true,
  "direction": "left",
  "target_detection": {
    "class_name": "person",
    "confidence": 0.85,
    "bbox": [100, 200, 300, 500],
    "centroid": [200, 350],
    "area": 80000,
    "area_ratio": 0.15
  },
  "tracking": {
    "object_id": 5,
    "velocity": [12.5, -8.3],  // pixels/second
    "frames_tracked": 42
  },
  "distance_ratio": 0.15,
  "reached": false
}
```

---

## 🔧 Tracking Algorithm Details

### **Centroid Distance Calculation:**
```python
distance = sqrt((x2 - x1)² + (y2 - y1)²)
```

### **Velocity Calculation:**
```python
vx = (current_x - previous_x) / time_delta
vy = (current_y - previous_y) / time_delta
```

### **Direction Zones:**
```python
if centroid_x < width * 0.35:
    direction = "left"
elif centroid_x > width * 0.65:
    direction = "right"
else:
    direction = "center"
```

---

## 🚀 Continuous Tracking Behavior

### **When Target Found:**
1. ✅ Calculate direction (left/center/right)
2. ✅ Calculate velocity (movement prediction)
3. ✅ Move robot toward target
4. ✅ Check if reached (area > 25%)
5. ✅ Repeat every 0.1 seconds

### **When Target Lost:**
1. 🔍 Rotate servo to scan (left → center → right)
2. 🔍 If still not found: Rotate robot 90°
3. 🔍 Move forward to explore new area
4. 🔍 Continue searching pattern

---

## ⚙️ Configuration Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_distance` | 100 pixels | Maximum distance for track matching |
| `max_history` | 30 frames | Position history length |
| `track_timeout` | 1.0 seconds | Time before removing lost track |
| `full_reset_timeout` | 5.0 seconds | Time before clearing all tracks |
| `command_cooldown` | 0.5 seconds | Minimum time between robot movements |
| `search_cooldown` | 1.0 seconds | Time between search actions |

---

## 🎬 Example Tracking Scenario

1. **Frame 1**: Person detected at (200, 300) → New track ID=1
2. **Frame 2**: Person at (205, 300) → Matched to ID=1 (distance=5px)
3. **Frame 3**: Person at (210, 300) → Matched to ID=1 (distance=5px)
4. **Frame 4**: Person at (215, 300) → Matched to ID=1 (distance=5px)
   - Velocity calculated: (5, 0) pixels/frame → Moving right
5. **Frame 5**: Person at (220, 300) → Matched to ID=1
   - Direction: "right" → Robot moves right
6. **Frame 10**: Person at (250, 300) → Still ID=1
   - Area ratio: 0.20 → Not reached yet
7. **Frame 20**: Person at (320, 300) → ID=1
   - Area ratio: 0.28 → **REACHED!** → Robot stops

---

## 🔍 Why This Approach?

1. **Simple & Fast**: Centroid distance is computationally efficient
2. **Robust**: Works well for single target tracking
3. **Real-time**: Processes 10 FPS (0.1s interval)
4. **Predictive**: Velocity helps anticipate movement
5. **Automatic**: Handles track creation/cleanup automatically

---

## 📝 Code Locations

- **Tracker Class**: `server.py` lines 171-261 (`ServerTracker`)
- **Stream Processing**: `server.py` lines 302-399 (`StreamProcessor._process_loop`)
- **Client Movement**: `client_rpi.py` lines 1229-1268 (`run_realtime`)

---

## 🎯 Summary

The tracking system:
1. ✅ Detects objects with YOLO
2. ✅ Matches detections to tracks using centroid distance
3. ✅ Maintains object IDs across frames
4. ✅ Calculates velocity for movement prediction
5. ✅ Determines direction for robot navigation
6. ✅ Continuously updates and moves robot toward target

This creates a **smooth, continuous tracking experience** where the robot follows objects in real-time!

