import cv2
import RPi.GPIO as GPIO
import time
import numpy as np

# ========== GPIO SETUP ==========
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# DC Motor pins (L298N Motor Driver)
MOTOR_LEFT_EN = 25
MOTOR_LEFT_IN1 = 23
MOTOR_LEFT_IN2 = 24

MOTOR_RIGHT_EN = 16
MOTOR_RIGHT_IN1 = 20
MOTOR_RIGHT_IN2 = 21

# Servo Motor pins (Robotic Arm)
SERVO_BASE = 17
SERVO_SHOULDER = 27
SERVO_ELBOW = 22
SERVO_GRIPPER = 18

# Setup DC Motor pins
motor_pins = [MOTOR_LEFT_EN, MOTOR_LEFT_IN1, MOTOR_LEFT_IN2,
              MOTOR_RIGHT_EN, MOTOR_RIGHT_IN1, MOTOR_RIGHT_IN2]
for pin in motor_pins:
    GPIO.setup(pin, GPIO.OUT)

# Setup Servo pins
servo_pins = [SERVO_BASE, SERVO_SHOULDER, SERVO_ELBOW, SERVO_GRIPPER]
servos = {}
for pin in servo_pins:
    GPIO.setup(pin, GPIO.OUT)
    servos[pin] = GPIO.PWM(pin, 50)
    servos[pin].start(0)

# DC Motor PWM
pwm_left = GPIO.PWM(MOTOR_LEFT_EN, 100)
pwm_right = GPIO.PWM(MOTOR_RIGHT_EN, 100)
pwm_left.start(0)
pwm_right.start(0)

# ========== MOTOR CONTROL ==========
def move_forward(speed=60):
    GPIO.output(MOTOR_LEFT_IN1, GPIO.HIGH)
    GPIO.output(MOTOR_LEFT_IN2, GPIO.LOW)
    GPIO.output(MOTOR_RIGHT_IN1, GPIO.HIGH)
    GPIO.output(MOTOR_RIGHT_IN2, GPIO.LOW)
    pwm_left.ChangeDutyCycle(speed)
    pwm_right.ChangeDutyCycle(speed)

def move_backward(speed=60):
    GPIO.output(MOTOR_LEFT_IN1, GPIO.LOW)
    GPIO.output(MOTOR_LEFT_IN2, GPIO.HIGH)
    GPIO.output(MOTOR_RIGHT_IN1, GPIO.LOW)
    GPIO.output(MOTOR_RIGHT_IN2, GPIO.HIGH)
    pwm_left.ChangeDutyCycle(speed)
    pwm_right.ChangeDutyCycle(speed)

def turn_left(speed=50):
    GPIO.output(MOTOR_LEFT_IN1, GPIO.LOW)
    GPIO.output(MOTOR_LEFT_IN2, GPIO.HIGH)
    GPIO.output(MOTOR_RIGHT_IN1, GPIO.HIGH)
    GPIO.output(MOTOR_RIGHT_IN2, GPIO.LOW)
    pwm_left.ChangeDutyCycle(speed)
    pwm_right.ChangeDutyCycle(speed)

def turn_right(speed=50):
    GPIO.output(MOTOR_LEFT_IN1, GPIO.HIGH)
    GPIO.output(MOTOR_LEFT_IN2, GPIO.LOW)
    GPIO.output(MOTOR_RIGHT_IN1, GPIO.LOW)
    GPIO.output(MOTOR_RIGHT_IN2, GPIO.HIGH)
    pwm_left.ChangeDutyCycle(speed)
    pwm_right.ChangeDutyCycle(speed)

def stop_motors():
    pwm_left.ChangeDutyCycle(0)
    pwm_right.ChangeDutyCycle(0)

# ========== SERVO CONTROL ==========
def set_servo_angle(pin, angle):
    duty = 2 + (angle / 18)
    servos[pin].ChangeDutyCycle(duty)
    time.sleep(0.3)
    servos[pin].ChangeDutyCycle(0)

def open_gripper():
    set_servo_angle(SERVO_GRIPPER, 90)

def close_gripper():
    set_servo_angle(SERVO_GRIPPER, 10)

def pickup_debris():
    print("Picking up debris...")
    set_servo_angle(SERVO_BASE, 90)
    set_servo_angle(SERVO_SHOULDER, 45)
    set_servo_angle(SERVO_ELBOW, 90)
    open_gripper()
    time.sleep(0.5)
    set_servo_angle(SERVO_SHOULDER, 20)
    close_gripper()
    time.sleep(0.5)
    set_servo_angle(SERVO_SHOULDER, 90)
    set_servo_angle(SERVO_BASE, 180)
    open_gripper()
    time.sleep(0.5)
    print("Debris placed in bin!")

# ========== OBJECT DETECTION ==========
def load_dataset(dataset_path='dataset/'):
    print("Loading debris dataset...")
    dataset = []
    import os
    if os.path.exists(dataset_path):
        for filename in os.listdir(dataset_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img = cv2.imread(dataset_path + filename)
                if img is not None:
                    dataset.append(img)
    print(f"Loaded {len(dataset)} images from dataset")
    return dataset

def detect_debris(frame, dataset):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for ref_img in dataset:
        gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        gray_ref = cv2.resize(gray_ref, (gray_frame.shape[1], gray_frame.shape[0]))
        diff = cv2.absdiff(gray_frame, gray_ref)
        similarity = 1 - (np.sum(diff) / (255 * diff.size))
        if similarity > 0.75:
            return True
    return False

def get_object_position(frame):
    height, width = frame.shape[:2]
    center_x = width // 2
    center_y = height // 2
    return center_x, center_y, width, height

# ========== MAIN PROGRAM ==========
def main():
    print("Automatic Debris Removal System Starting...")
    print("Developed by: V. Soma Sekhar — Sathyabama University")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not found!")
        return

    dataset = load_dataset()
    print("System ready. Scanning for debris...")

    STOP_DISTANCE = 15
    debris_collected = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera error!")
                break

            debris_detected = detect_debris(frame, dataset)

            if debris_detected:
                print(f"Debris detected! Total collected: {debris_collected}")
                obj_x, obj_y, frame_w, frame_h = get_object_position(frame)

                if obj_x < frame_w // 3:
                    print("Turning left to align...")
                    turn_left()
                    time.sleep(0.3)
                elif obj_x > 2 * frame_w // 3:
                    print("Turning right to align...")
                    turn_right()
                    time.sleep(0.3)
                else:
                    print("Moving forward toward debris...")
                    move_forward()
                    time.sleep(0.5)

                stop_motors()
                time.sleep(0.2)
                pickup_debris()
                debris_collected += 1
                print(f"Debris collected! Total: {debris_collected}")
                time.sleep(1)
            else:
                move_forward(speed=40)
                time.sleep(0.3)
                stop_motors()

            cv2.putText(frame, f"Debris Detected: {debris_detected}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0) if debris_detected else (0, 0, 255), 2)
            cv2.putText(frame, f"Collected: {debris_collected}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.imshow('Debris Removal System', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("System stopped by user")
                break

    except KeyboardInterrupt:
        print("System interrupted!")

    finally:
        stop_motors()
        for pin in servo_pins:
            servos[pin].stop()
        pwm_left.stop()
        pwm_right.stop()
        cap.release()
        cv2.destroyAllWindows()
        GPIO.cleanup()
        print(f"Session complete. Total debris collected: {debris_collected}")

if __name__ == "__main__":
    main()
