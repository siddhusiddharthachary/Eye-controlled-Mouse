import cv2
import mediapipe as mp
import pyautogui

# =====================================
# SETTINGS
# =====================================

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

SMOOTHING = 0.2

# =====================================
# SCREEN SIZE
# =====================================

screen_w, screen_h = pyautogui.size()

# =====================================
# WEBCAM
# =====================================

cap = cv2.VideoCapture(0)

# =====================================
# MEDIAPIPE HANDS
# =====================================

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# =====================================
# SMOOTHING VARIABLES
# =====================================

prev_x = 0
prev_y = 0

# =====================================
# MAIN LOOP
# =====================================

while True:

    success, frame = cap.read()

    if not success:
        break

    # Flip frame
    frame = cv2.flip(frame, 1)

    frame_h, frame_w, _ = frame.shape

    # RGB conversion
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw hand landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # =====================================
            # INDEX FINGER TIP
            # =====================================

            index_tip = hand_landmarks.landmark[8]

            x = int(index_tip.x * frame_w)
            y = int(index_tip.y * frame_h)

            # Draw fingertip
            cv2.circle(frame, (x, y), 15, (0, 255, 0), -1)

            # =====================================
            # SCREEN MAPPING
            # =====================================

            screen_x = screen_w * index_tip.x
            screen_y = screen_h * index_tip.y

            # =====================================
            # SMOOTHING
            # =====================================

            curr_x = prev_x + (screen_x - prev_x) * SMOOTHING
            curr_y = prev_y + (screen_y - prev_y) * SMOOTHING

            # Move cursor
            pyautogui.moveTo(curr_x, curr_y)

            prev_x = curr_x
            prev_y = curr_y

            # =====================================
            # DEBUG TEXT
            # =====================================

            cv2.putText(
                frame,
                f"X:{int(screen_x)} Y:{int(screen_y)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )

    # =====================================
    # SHOW WINDOW
    # =====================================

    cv2.imshow("AI Virtual Mouse", frame)

    # Exit with Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =====================================
# CLEANUP
# =====================================

cap.release()
cv2.destroyAllWindows()