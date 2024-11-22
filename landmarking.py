import cv2
import mediapipe as mp

# Initialize Mediapipe solutions
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
face = mp_face.FaceMesh(min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define indices for key facial landmarks (eyebrows, eyes, mouth)
KEY_FACE_LANDMARKS = [
    33,  # Left eye
    263, # Right eye
    61,  # Mouth (left)
    291, # Mouth (right)
    159, # Left eyebrow
    386, # Right eyebrow
    13,  # Nose tip
    14   # Lower lip
]

# Capture video feed
cap = cv2.VideoCapture(1)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb_frame)
    face_results = face.process(rgb_frame)

    # Draw hand landmarks
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Draw selected face landmarks
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            for idx in KEY_FACE_LANDMARKS:
                landmark = face_landmarks.landmark[idx]
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow('ASL Translator', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
