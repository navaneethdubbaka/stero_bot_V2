#include "AFMotor.h"
#include <Servo.h>

#define SERVO_PIN 10     // Servo signal

// ====================
// Motor Definitions
// ====================
AF_DCMotor motor1(1, MOTOR12_64KHZ); // Left motor (M2 on shield)
AF_DCMotor motor2(3, MOTOR12_64KHZ); // Right motor (M3 on shield)

// ====================
// Global Objects & Variables
// ====================
Servo myservo;
int currentServoAngle = 90;      // Servo starts centered at 90°

// Motor speed constants
const int NORMAL_SPEED = 160;
const int TURN_SPEED_SLOW = 200;
const int TURN_SPEED_FAST = 240;

// ====================
// Function Prototypes
// ====================
void slowServoWrite(int targetAngle);
void moveForward();
void turnLeft();
void turnRight();
void turnFarLeft();
void turnFarRight();
void turnCenterLeft();
void turnCenterRight();
void stopMotors();
void rotateBack();
void processCommand(String cmd);
void handshakeRotation(int targetAngle, String angleName);

// ====================
// Setup Function
// ====================
void setup() {
  Serial.begin(9600);
  Serial.println("Arduino Robot Initialized via Serial Commands");
  
  // Attach the servo and center it at 90°
  myservo.attach(SERVO_PIN);
  slowServoWrite(90);
  
  // Set default motor speeds
  motor1.setSpeed(NORMAL_SPEED);
  motor2.setSpeed(NORMAL_SPEED);
  
  Serial.println("Ready for commands");
}

// ====================
// Main Loop
// ====================
void loop() {
  // Process incoming Serial commands from the Raspberry Pi
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim(); // Remove any whitespace/newlines
    if(command.length() > 0) {
      Serial.print("Received command: ");
      Serial.println(command);
      processCommand(command);
    }
  }
  
  delay(50); // Short delay to avoid flooding
}

// ====================
// Helper Functions
// ====================

// Gradually move servo from currentServoAngle to targetAngle
void slowServoWrite(int targetAngle) {
  int stepDelay = 3; // Delay in ms per degree
  if (currentServoAngle < targetAngle) {
    for (int pos = currentServoAngle; pos <= targetAngle; pos++) {
      myservo.write(pos);
      delay(stepDelay);
    }
  } else if (currentServoAngle > targetAngle) {
    for (int pos = currentServoAngle; pos >= targetAngle; pos--) {
      myservo.write(pos);
      delay(stepDelay);
    }
  }
  currentServoAngle = targetAngle;
}

// Performs a handshake for servo positioning.
// Rotates the servo, sends "ROTATION_ACK", and waits for "PHOTO_ACK".
void handshakeRotation(int targetAngle, String angleName) {
  Serial.print("Setting servo to ");
  Serial.print(angleName);
  Serial.print(" (");
  Serial.print(targetAngle);
  Serial.println("°)");
  
  slowServoWrite(targetAngle);
  delay(500);  // Allow time for servo to complete motion
  
  // Send rotation acknowledgement to Raspberry Pi
  Serial.println("ROTATION_ACK");
  
  // Wait for PHOTO_ACK from Raspberry Pi before proceeding
  unsigned long startTime = millis();
  unsigned long timeout = 3000; // 3 second timeout
  
  while (millis() - startTime < timeout) {
    if (Serial.available() > 0) {
      String ack = Serial.readStringUntil('\n');
      ack.trim();
      if (ack == "PHOTO_ACK") {
        Serial.println("PHOTO_ACK received. Proceeding.");
        return;
      }
    }
    delay(10);
  }
  
  Serial.println("PHOTO_ACK timeout - proceeding anyway");
}

// ====================
// Movement Functions
// ====================

void moveForward() {
  Serial.println("Moving forward");
  motor1.setSpeed(NORMAL_SPEED);
  motor2.setSpeed(NORMAL_SPEED);
  motor1.run(FORWARD);
  motor2.run(FORWARD);
}

void turnLeft() {
  Serial.println("Turning left");
  motor1.setSpeed(TURN_SPEED_SLOW);
  motor2.setSpeed(TURN_SPEED_FAST);
  motor1.run(BACKWARD);
  motor2.run(FORWARD);
  delay(550);  // Adjust delay for desired turn angle
  motor1.setSpeed(NORMAL_SPEED);
  motor2.setSpeed(NORMAL_SPEED);
}

void turnRight() {
  Serial.println("Turning right");
  motor1.setSpeed(TURN_SPEED_FAST);
  motor2.setSpeed(TURN_SPEED_SLOW);
  motor1.run(FORWARD);
  motor2.run(BACKWARD);
  delay(550);
  motor1.setSpeed(NORMAL_SPEED);
  motor2.setSpeed(NORMAL_SPEED);
}

void turnFarLeft() {
  Serial.println("Turning far left");
  motor1.setSpeed(TURN_SPEED_SLOW);
  motor2.setSpeed(TURN_SPEED_FAST);
  motor1.run(BACKWARD);
  motor2.run(FORWARD);
  delay(1100);  // Longer delay for far turn
  motor1.setSpeed(NORMAL_SPEED);
  motor2.setSpeed(NORMAL_SPEED);
}

void turnFarRight() {
  Serial.println("Turning far right");
  motor1.setSpeed(TURN_SPEED_FAST);
  motor2.setSpeed(TURN_SPEED_SLOW);
  motor1.run(FORWARD);
  motor2.run(BACKWARD);
  delay(1100);  // Longer delay for far turn
  motor1.setSpeed(NORMAL_SPEED);
  motor2.setSpeed(NORMAL_SPEED);
}

void turnCenterLeft() {
  Serial.println("Turning center-left");
  motor1.setSpeed(TURN_SPEED_SLOW);
  motor2.setSpeed(TURN_SPEED_FAST);
  motor1.run(BACKWARD);
  motor2.run(FORWARD);
  delay(275);  // Shorter delay for center turn
  motor1.setSpeed(NORMAL_SPEED);
  motor2.setSpeed(NORMAL_SPEED);
}

void turnCenterRight() {
  Serial.println("Turning center-right");
  motor1.setSpeed(TURN_SPEED_FAST);
  motor2.setSpeed(TURN_SPEED_SLOW);
  motor1.run(FORWARD);
  motor2.run(BACKWARD);
  delay(275);  // Shorter delay for center turn
  motor1.setSpeed(NORMAL_SPEED);
  motor2.setSpeed(NORMAL_SPEED);
}

void stopMotors() {
  Serial.println("Stopping motors");
  motor1.run(RELEASE);
  motor2.run(RELEASE);
}

// Rotates the robot 180° (back rotation) using a spin turn
void rotateBack() {
  Serial.println("Performing 180-degree back rotation");
  motor1.setSpeed(NORMAL_SPEED);
  motor2.setSpeed(NORMAL_SPEED);
  // Spin in place: left motor forward, right motor backward
  motor1.run(FORWARD);
  motor2.run(BACKWARD);
  delay(1550);  // Adjust delay as necessary for a 180° rotation
  stopMotors();
}

// ====================
// Command Processing
// ====================

void processCommand(String cmd) {
  // Servo positioning commands with handshake
  if (cmd == "SC") {
    handshakeRotation(90, "CENTER");
  } else if (cmd == "SL") {
    handshakeRotation(180, "LEFT");
  } else if (cmd == "SR") {
    handshakeRotation(0, "RIGHT");
  }
  
  // Movement commands (from Python client_rpi.py COMMANDS dictionary)
  else if (cmd == "MC") {
    Serial.println("Driving forward (center)");
    moveForward();
  } else if (cmd == "ML") {
    Serial.println("Turning left then moving forward");
    turnLeft();
    delay(200);
    moveForward();
  } else if (cmd == "MR") {
    Serial.println("Turning right then moving forward");
    turnRight();
    delay(200);
    moveForward();
  } else if (cmd == "MFL") {
    Serial.println("Turning far left then moving forward");
    turnFarLeft();
    delay(200);
    moveForward();
  } else if (cmd == "MFR") {
    Serial.println("Turning far right then moving forward");
    turnFarRight();
    delay(200);
    moveForward();
  } else if (cmd == "MCL") {
    Serial.println("Turning center-left then moving forward");
    turnCenterLeft();
    delay(200);
    moveForward();
  } else if (cmd == "MCR") {
    Serial.println("Turning center-right then moving forward");
    turnCenterRight();
    delay(200);
    moveForward();
  }
  
  // Stop command
  else if (cmd == "ST") {
    stopMotors();
  }
  
  // Back rotation command
  else if (cmd == "RB") {
    Serial.println("Back rotation command received");
    rotateBack();
  }
  
  // Legacy manual control commands (kept for backward compatibility)
  else if (cmd == "w") {
    Serial.println("Driving forward (center)");
    moveForward();
  } else if (cmd == "a") {
    Serial.println("Turning left");
    turnLeft();
    delay(200);
  } else if (cmd == "d") {
    Serial.println("Turning right");
    turnRight();
    delay(200);
  } else if (cmd == "s") {
    Serial.println("Back rotation command received");
    rotateBack();
  } else if (cmd == "z") {
    Serial.println("Stopped");
    stopMotors();
  }
  
  // Unknown command
  else {
    Serial.print("Unknown command: ");
    Serial.println(cmd);
  }
}

