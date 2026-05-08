import cv2
import mediapipe as mp

# Webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Left iris landmark indices
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

while True:
    success, frame = cap.read()

    if not success:
        break

    # Flip frame like mirror
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = face_mesh.process(rgb_frame)

    frame_h, frame_w, _ = frame.shape

    # If face detected
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

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

                # Calculate center
                center_x = int(sum([p[0] for p in iris_points]) / len(iris_points))
                center_y = int(sum([p[1] for p in iris_points]) / len(iris_points))

                # Draw center point
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

                print(f"Iris Center: {center_x}, {center_y}")

    # Show output
    cv2.imshow("Eye Controlled Mouse", frame)

    # Exit on Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()