import cv2
import mediapipe as mp
import pyautogui

# Webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Get screen size
screen_w, screen_h = pyautogui.size()

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Iris landmark indices
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Smoothing variables
prev_x = 0
prev_y = 0

# Smoothing factor
smoothening = 0.2

while True:
    success, frame = cap.read()

    if not success:
        break

    # Mirror effect
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = face_mesh.process(rgb_frame)

    frame_h, frame_w, _ = frame.shape

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            eye_centers = []

            # Process both eyes
            for iris_indices in [LEFT_IRIS, RIGHT_IRIS]:

                iris_points = []

                for idx in iris_indices:

                    landmark = face_landmarks.landmark[idx]

                    x = int(landmark.x * frame_w)
                    y = int(landmark.y * frame_h)

                    iris_points.append((x, y))

                    # Draw iris points
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                # Calculate eye center
                center_x = int(sum([p[0] for p in iris_points]) / len(iris_points))
                center_y = int(sum([p[1] for p in iris_points]) / len(iris_points))

                eye_centers.append((center_x, center_y))

                # Draw center point
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

            # Average both eyes
            avg_x = int(sum([p[0] for p in eye_centers]) / len(eye_centers))
            avg_y = int(sum([p[1] for p in eye_centers]) / len(eye_centers))

            # Draw combined center
            cv2.circle(frame, (avg_x, avg_y), 8, (255, 0, 0), -1)

            # Convert to screen coordinates
            screen_x = screen_w * (avg_x / frame_w)
            screen_y = screen_h * (avg_y / frame_h)

            # Smooth mouse movement
            curr_x = prev_x + (screen_x - prev_x) * smoothening
            curr_y = prev_y + (screen_y - prev_y) * smoothening

            # Move mouse
            pyautogui.moveTo(curr_x, curr_y)

            # Update previous position
            prev_x = curr_x
            prev_y = curr_y

    # Show output
    cv2.imshow("Eye Controlled Mouse", frame)

    # Exit on Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()