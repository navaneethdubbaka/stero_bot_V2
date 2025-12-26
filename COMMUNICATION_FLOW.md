# Communication Flow: Server ↔ Client

## Overview

The system uses **HTTP polling** for communication between:
- **Server (Laptop)**: `server.py` - FastAPI server running on your laptop
- **Client (Raspberry Pi)**: `client_rpi.py` - Robot controller running on Raspberry Pi

---

## How Commands Flow from UI to Robot

### Step-by-Step Flow:

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│   Web Browser   │         │   Server (Laptop) │         │ Client (Rasp Pi)│
│  (Your Browser) │         │   server.py:8000  │         │ client_rpi.py   │
└────────┬────────┘         └─────────┬─────────┘         └────────┬────────┘
         │                             │                            │
         │ 1. User clicks button       │                            │
         │    (e.g., "Forward")        │                            │
         ├────────────────────────────>│                            │
         │                             │                            │
         │ 2. POST /control/command    │                            │
         │    command="forward"         │                            │
         │                             │                            │
         │                             │ 3. Add to queue            │
         │                             │    manual_control_queue    │
         │                             │    .append("forward")      │
         │                             │                            │
         │                             │                            │
         │                             │<───────────────────────────┤
         │                             │ 4. GET /control/poll      │
         │                             │    (every 100ms)           │
         │                             │                            │
         │                             │ 5. Return queued command   │
         │                             │    {"command": "forward",  │
         │                             │     "has_command": true}   │
         │                             ├───────────────────────────>│
         │                             │                            │
         │                             │                            │ 6. Execute command
         │                             │                            │    motors.move("center")
         │                             │                            │
```

---

## Detailed Communication Mechanism

### 1. **Command Queue System (Server Side)**

On your laptop, `server.py` maintains a **command queue**:

```python
# In server.py
manual_control_queue = []  # List to store commands
manual_control_lock = threading.Lock()  # Thread-safe access
```

**When UI sends a command:**
```python
@app.post("/control/command")
async def send_control_command(command: str = Form(...)):
    with manual_control_lock:
        manual_control_queue.append(command)  # Add to queue
    return {"success": True, "message": "Command queued"}
```

### 2. **Polling Mechanism (Client Side)**

On Raspberry Pi, `client_rpi.py` **polls** the server every 100ms:

```python
# In client_rpi.py - run_realtime() or run_manual_control()
while self.running:
    # Poll server for commands
    manual_command = self.client.poll_control_command()
    
    if manual_command:
        # Execute the command
        self.motors.move(manual_command)
    
    time.sleep(0.1)  # Poll every 100ms
```

**The polling function:**
```python
def poll_control_command(self) -> Optional[str]:
    """Poll server for manual control command."""
    try:
        # HTTP GET request to server
        resp = self.session.get(f"{self.server_url}/control/poll", timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("has_command"):
                return data.get("command")  # Return command if available
        return None  # No command available
    except Exception:
        return None
```

### 3. **Server Response (Poll Endpoint)**

When the client polls, the server checks the queue:

```python
@app.get("/control/poll")
async def poll_control_command():
    """Poll for manual control commands (used by Pi client)."""
    with manual_control_lock:
        if manual_control_queue:
            cmd = manual_control_queue.pop(0)  # Remove first command
            return {"command": cmd, "has_command": True}
        return {"command": None, "has_command": False}  # No commands
```

---

## Network Communication Details

### HTTP Requests Over Network

**Client → Server (Polling):**
```
GET http://LAPTOP_IP:8000/control/poll
```

**Server → Client (Response):**
```json
{
  "command": "forward",
  "has_command": true
}
```

Or if no command:
```json
{
  "command": null,
  "has_command": false
}
```

### Network Requirements

1. **Both devices on same network** (WiFi/LAN)
2. **Server IP accessible** from Raspberry Pi
3. **Port 8000 open** on laptop (server port)
4. **Firewall allows connections** (if enabled)

---

## Example: Complete Command Flow

### Scenario: User clicks "Forward" button in UI

**1. Browser → Server:**
```javascript
// In server.py UI JavaScript
async function sendCommand(command) {
    const formData = new FormData();
    formData.append('command', command);
    
    await fetch('http://LAPTOP_IP:8000/control/command', {
        method: 'POST',
        body: formData
    });
}

// User clicks "Forward" button
sendCommand('forward');
```

**2. Server stores command:**
```python
# In server.py
manual_control_queue = ['forward']  # Command added to queue
```

**3. Client polls (happens every 100ms):**
```python
# In client_rpi.py
# HTTP GET request:
GET http://192.168.1.50:8000/control/poll

# Server responds:
{"command": "forward", "has_command": true}
```

**4. Client executes:**
```python
# In client_rpi.py
if manual_command == "forward":
    self.motors.move("center", 0.5)  # Move robot forward
    print("  ✓ Moving forward")
```

**5. Command removed from queue:**
```python
# Server removes command after sending
manual_control_queue = []  # Queue is now empty
```

---

## Why Polling Instead of Push?

**Polling (Current Method):**
- ✅ Simple to implement
- ✅ Works through firewalls/NAT
- ✅ No need for persistent connections
- ✅ Client controls request rate
- ⚠️ Slight delay (up to 100ms)

**Push (Alternative - Not Used):**
- Would require WebSockets or persistent connection
- More complex setup
- May have firewall issues
- Lower latency

**For this use case, polling every 100ms is fast enough and simpler.**

---

## Network Configuration

### Finding IP Addresses

**On Laptop (Server):**
```bash
# Windows
ipconfig
# Look for IPv4 Address (e.g., 192.168.1.50)

# Linux/Mac
ifconfig
# or
ip addr show
```

**On Raspberry Pi (Client):**
```bash
hostname -I
# or
ip addr show
```

### Starting the System

**1. Start Server (Laptop):**
```bash
python server.py
# Server runs on http://0.0.0.0:8000
# Accessible at http://LAPTOP_IP:8000
```

**2. Start Client (Raspberry Pi):**
```bash
python client_rpi.py --server http://LAPTOP_IP:8000
# Example: python client_rpi.py --server http://192.168.1.50:8000
```

**3. Access UI (Browser):**
```
http://LAPTOP_IP:8000/ui
# Example: http://192.168.1.50:8000/ui
```

---

## Troubleshooting Communication Issues

### Client Can't Connect to Server

**Check:**
1. Server is running: `python server.py`
2. Correct IP address: Use laptop's actual IP, not `localhost`
3. Same network: Both devices on same WiFi/LAN
4. Firewall: Allow port 8000 on laptop
5. Test connection:
   ```bash
   # From Raspberry Pi
   curl http://LAPTOP_IP:8000/
   ```

### Commands Not Received

**Check:**
1. Client is polling: Look for `poll_control_command()` calls in logs
2. Server queue: Commands should be added to `manual_control_queue`
3. Network latency: Check if requests are timing out
4. Server logs: Check for errors in server console

### High Latency

**Solutions:**
1. Reduce polling interval (currently 100ms)
2. Check network speed
3. Ensure devices on same network segment
4. Check for network congestion

---

## Summary

**Communication Method:** HTTP Polling
- Client polls server every 100ms
- Server maintains command queue
- Commands are FIFO (First In, First Out)
- Works over standard HTTP (port 8000)
- No special network setup required

**Flow:**
1. UI sends command → Server queue
2. Client polls → Server responds with command
3. Client executes → Robot moves
4. Process repeats

This design allows the robot to be controlled remotely from any device with a web browser, as long as it can access the server's IP address.

