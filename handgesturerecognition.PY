import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Capture Webcam
cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()  # Get screen size

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror effect
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract Landmark Positions
            landmarks = []
            h, w, c = frame.shape
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append([cx, cy])

            if landmarks:
                thumb_tip = landmarks[4]
                index_tip = landmarks[8]
                middle_tip = landmarks[12]
                ring_tip = landmarks[16]
                pinky_tip = landmarks[20]
                wrist = landmarks[0]

                # **1️⃣ Move Mouse Cursor**
                index_x, index_y = index_tip[0], index_tip[1]
                pyautogui.moveTo(index_x * screen_width // w, index_y * screen_height // h, duration=0.1)

                # **2️⃣ Click Mouse when Fist is Detected**
                if (
                    index_tip[1] > landmarks[6][1] and
                    middle_tip[1] > landmarks[10][1] and
                    ring_tip[1] > landmarks[14][1] and
                    pinky_tip[1] > landmarks[18][1]
                ):
                    pyautogui.click()
                    cv2.putText(frame, "Click!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                # **3️⃣ Volume Control**
                # Volume Up when Thumb Up
                if (
                    thumb_tip[1] < wrist[1] and
                    index_tip[1] > landmarks[6][1] and
                    middle_tip[1] > landmarks[10][1] and
                    ring_tip[1] > landmarks[14][1] and
                    pinky_tip[1] > landmarks[18][1]
                ):
                    pyautogui.press("volumeup")
                    cv2.putText(frame, "Volume Up", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

                # Volume Down when Thumb Down
                if (
                    thumb_tip[1] > wrist[1] and
                    index_tip[1] > landmarks[6][1] and
                    middle_tip[1] > landmarks[10][1] and
                    ring_tip[1] > landmarks[14][1] and
                    pinky_tip[1] > landmarks[18][1]
                ):
                    pyautogui.press("volumedown")
                    cv2.putText(frame, "Volume Down", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

                # **4️⃣ Open Browser with "Peace Sign"**
                if (
                    index_tip[1] < wrist[1] and
                    middle_tip[1] < wrist[1] and
                    ring_tip[1] > wrist[1] and
                    pinky_tip[1] > wrist[1]
                ):
                    pyautogui.hotkey('ctrl', 't')  # Open new tab in browser
                    cv2.putText(frame, "Open Browser", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Show the frame
    cv2.imshow("Hand Gesture Control", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
