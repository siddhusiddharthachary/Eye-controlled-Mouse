import cv2
import mediapipe as mp
import pyautogui

# Webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Screen size
screen_w, screen_h = pyautogui.size()

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# Iris landmarks
LEFT_IRIS = [474, 475, 476, 477]

LEFT_EYE_LEFT_CORNER = 263
LEFT_EYE_RIGHT_CORNER = 362

# Smoothing
prev_x = 0
smoothening = 0.2

while True:
    success, frame = cap.read()

    if not success:
        break

    # Mirror effect
    frame = cv2.flip(frame, 1)

    # RGB conversion
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = face_mesh.process(rgb_frame)

    frame_h, frame_w, _ = frame.shape

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            # ----------------------------
            # IRIS CENTER
            # ----------------------------

            iris_points = []

            for idx in LEFT_IRIS:

                landmark = face_landmarks.landmark[idx]

                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)

                iris_points.append((x, y))

                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            iris_x = int(sum([p[0] for p in iris_points]) / len(iris_points))
            iris_y = int(sum([p[1] for p in iris_points]) / len(iris_points))

            cv2.circle(frame, (iris_x, iris_y), 5, (0, 0, 255), -1)

            # ----------------------------
            # EYE CORNERS
            # ----------------------------

            left_corner = face_landmarks.landmark[LEFT_EYE_LEFT_CORNER]
            right_corner = face_landmarks.landmark[LEFT_EYE_RIGHT_CORNER]

            x1 = int(left_corner.x * frame_w)
            x2 = int(right_corner.x * frame_w)
            left_x = min(x1, x2)
            right_x = max(x1, x2)

            # Draw corners
            cv2.circle(frame, (left_x, iris_y), 5, (255, 0, 0), -1)
            cv2.circle(frame, (right_x, iris_y), 5, (255, 0, 0), -1)

            # ----------------------------
            # RELATIVE POSITION
            # ----------------------------

            eye_width = right_x - left_x

            if eye_width != 0:

                ratio = (iris_x - left_x) / eye_width
                ratio = (ratio - 0.5) * 5 + 0.5
                ratio = max(0, min(ratio, 1))
                
                

                # Convert to screen coordinates
                screen_x = ratio * screen_w * 2
                screen_x = max(0, min(screen_x, screen_w))

                # Smooth movement
                curr_x = prev_x + (screen_x - prev_x) * smoothening
                print(
                    f"IrisX: {iris_x}, "
                    f"LeftX: {left_x}, "
                    f"RightX: {right_x}, "
                    f"EyeWidth: {eye_width}"
                )
                print(f"Ratio: {ratio:.2f}")

                # Move mouse horizontally
                pyautogui.moveTo(curr_x, screen_h / 2)

                prev_x = curr_x

                # Debug text
                cv2.putText(
                    frame,
                    f"Ratio: {ratio:.2f}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2
                )

    # Show frame
    cv2.imshow("Relative Eye Tracking", frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
