import cv2
import mediapipe as mp

# Webcam
cap = cv2.VideoCapture(0)

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1
)

# Drawing utilities
mp_draw = mp.solutions.drawing_utils

while True:
    success, frame = cap.read()

    if not success:
        break

    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = face_mesh.process(rgb_frame)

    # If face detected
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION
            )

    # Show frame
    cv2.imshow("Face Mesh", frame)

    # Exit on Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()