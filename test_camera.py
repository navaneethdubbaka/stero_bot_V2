#!/usr/bin/env python3
"""
Camera Test Utility
===================
Tests camera access and displays diagnostic information.
"""

import cv2
import os
import glob
import sys

def list_video_devices():
    """List all available video devices."""
    devices = []
    for device_path in glob.glob('/dev/video*'):
        if os.path.exists(device_path):
            devices.append(device_path)
    return sorted(devices)

def check_permissions():
    """Check camera permissions."""
    devices = list_video_devices()
    if not devices:
        return False, "No video devices found"
    
    permissions = []
    for device in devices:
        stat = os.stat(device)
        perms = oct(stat.st_mode)[-3:]
        permissions.append(f"{device}: {perms}")
    
    return True, "\n".join(permissions)

def test_camera_by_path(device_path):
    """Test camera by device path."""
    print(f"\nTesting {device_path}...")
    try:
        cap = cv2.VideoCapture(device_path)
        if not cap.isOpened():
            print(f"  ❌ Failed to open {device_path}")
            return False
        
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"  ❌ Opened but cannot read frames")
            cap.release()
            return False
        
        print(f"  ✓ Success! Frame shape: {frame.shape}")
        print(f"    Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        cap.release()
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_camera_by_index(index):
    """Test camera by index."""
    print(f"\nTesting camera index {index}...")
    try:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            print(f"  ❌ Failed to open index {index}")
            return False
        
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"  ❌ Opened but cannot read frames")
            cap.release()
            return False
        
        print(f"  ✓ Success! Frame shape: {frame.shape}")
        print(f"    Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        cap.release()
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def main():
    print("=" * 60)
    print("  CAMERA DIAGNOSTIC TOOL")
    print("=" * 60)
    
    # Check video devices
    print("\n[1] Checking video devices...")
    devices = list_video_devices()
    if devices:
        print(f"✓ Found {len(devices)} device(s):")
        for device in devices:
            print(f"  - {device}")
    else:
        print("❌ No video devices found!")
        print("\nTroubleshooting:")
        print("  1. Check camera is connected: lsusb")
        print("  2. Check if camera is detected: dmesg | tail -20")
        return
    
    # Check permissions
    print("\n[2] Checking permissions...")
    has_perms, perm_info = check_permissions()
    if has_perms:
        print(perm_info)
        # Check if user is in video group
        import grp
        try:
            video_gid = grp.getgrnam('video').gr_gid
            user_gids = os.getgroups()
            if video_gid in user_gids:
                print("✓ User is in 'video' group")
            else:
                print("⚠ User is NOT in 'video' group")
                print("  Fix: sudo usermod -a -G video $USER")
                print("  Then logout and login again")
        except KeyError:
            print("⚠ 'video' group not found")
    else:
        print(perm_info)
    
    # Test cameras
    print("\n[3] Testing cameras...")
    working_cameras = []
    
    # Test by device path
    for device in devices:
        if test_camera_by_path(device):
            working_cameras.append(device)
    
    # Test by index
    for idx in range(5):
        if test_camera_by_index(idx):
            working_cameras.append(f"index {idx}")
    
    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    if working_cameras:
        print(f"✓ Found {len(working_cameras)} working camera(s):")
        for cam in working_cameras:
            print(f"  - {cam}")
        print("\nYou can use:")
        if any('/dev/video' in c for c in working_cameras):
            device = [c for c in working_cameras if '/dev/video' in c][0]
            print(f"  Device path: {device}")
        if any('index' in c for c in working_cameras):
            idx = [c for c in working_cameras if 'index' in c][0].split()[-1]
            print(f"  Camera index: {idx}")
    else:
        print("❌ No working cameras found!")
        print("\nTroubleshooting:")
        print("  1. Check USB connection")
        print("  2. Try different USB port")
        print("  3. Check permissions: sudo usermod -a -G video $USER")
        print("  4. Check if camera is in use: lsof | grep video")
        print("  5. Try: v4l2-ctl --list-devices")
    
    print("=" * 60)

if __name__ == "__main__":
    main()

