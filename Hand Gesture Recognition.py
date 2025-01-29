import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Function to determine if the hand is open or closed
def is_hand_open(landmarks):
    thumb_tip = landmarks[4].y
    index_tip = landmarks[8].y
    middle_tip = landmarks[12].y
    
    # Compare positions to determine if hand is open
    return thumb_tip < index_tip and thumb_tip < middle_tip

# Function to detect the number being shown with fingers
def count_fingers(landmarks):
    tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    count = 0
    
    for tip in tips[1:]:  # Skip thumb for now
        if landmarks[tip].y < landmarks[tip - 2].y:  # Compare tip with lower joint
            count += 1

    # Special handling for thumb
    if landmarks[4].x > landmarks[3].x:  # Right hand
        count += 1
    elif landmarks[4].x < landmarks[3].x:  # Left hand
        count += 1

    return count

# Function to draw a bounding box around the hand
def draw_bounding_box(frame, landmarks):
    h, w, _ = frame.shape
    x_min = int(min([lm.x for lm in landmarks]) * w)
    y_min = int(min([lm.y for lm in landmarks]) * h)
    x_max = int(max([lm.x for lm in landmarks]) * w)
    y_max = int(max([lm.y for lm in landmarks]) * h)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

# Start video capture
cap = cv2.VideoCapture(0)

# Initialize FPS calculation
prev_time = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and detect hand landmarks
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Determine hand gesture (open or closed)
            if is_hand_open(hand_landmarks.landmark):
                gesture = " Open"
            else:
                gesture = "Closed"

            # Count the number of fingers
            finger_count = count_fingers(hand_landmarks.landmark)

            # Draw bounding box
            draw_bounding_box(frame, hand_landmarks.landmark)

            # Display the gesture and finger count on the video
            cv2.putText(frame, gesture, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Fingers: {finger_count}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Calculate and display FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the number of hands detected
    hand_count = len(result.multi_hand_landmarks) if result.multi_hand_landmarks else 0
    cv2.putText(frame, f'Hands Detected: {hand_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Exit with 'q' key
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
