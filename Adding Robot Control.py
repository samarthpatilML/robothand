import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

# Initialize variables for drawing
drawing = False  # Track if the user is drawing
prev_x, prev_y = None, None  # Previous coordinates for drawing
canvas = None  # Canvas to draw on

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the frame horizontally for a mirrored view
    frame = cv2.flip(frame, 1)

    # Initialize canvas if not already done
    if canvas is None:
        canvas = np.zeros_like(frame)

    # Convert the image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and detect hand landmarks
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the index finger tip coordinates
            index_finger_tip = hand_landmarks.landmark[8]
            h, w, _ = frame.shape
            x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            # Determine the number of fingers raised
            fingers = [
                hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y,  # Index finger
                hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y,  # Middle finger
                hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y,  # Ring finger
                hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y   # Pinky finger
            ]

            thumb_up = hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x if hand_landmarks.landmark[4].y < hand_landmarks.landmark[3].y else False

            # Count the number of raised fingers
            fingers_raised = sum(fingers) + (1 if thumb_up else 0)

            if fingers_raised == 1:  # One finger raised, enable drawing
                drawing = True
            elif fingers_raised == 5:  # All fingers raised, erase canvas
                canvas = np.zeros_like(frame)
                drawing = False
            else:
                drawing = False

            if drawing:
                if prev_x is not None and prev_y is not None:
                    # Ensure the line doesn't skip by verifying valid previous points
                    cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 0, 0), 5)
                prev_x, prev_y = x, y
            else:
                prev_x, prev_y = None, None

    else:
        prev_x, prev_y = None, None  # Reset when no hand is detected

    # Combine canvas and original frame
    blended_frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Display the resulting frame
    cv2.imshow('Drawing with Finger', blended_frame)

    # Exit with 'q' key
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
