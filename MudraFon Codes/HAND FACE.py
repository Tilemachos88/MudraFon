import mediapipe as mp
import cv2
from pythonosc import udp_client

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands

# Define the OSC client
ip = "127.0.0.1"  # i.e., localhost
sendPort = 32000  # UDP port
client = udp_client.SimpleUDPClient(ip, sendPort)

# Define the function for mapping coordinates to MIDI values
def map_to_midi(value, min_val, max_val):
    return int((value - min_val) / (max_val - min_val) * 127)

# Define the landmarks that correspond to each fingertip
tip_landmark_ids = [4, 8, 12, 16, 20]

# Define the threshold for detecting a finger as "up" (i.e., extended)
FINGER_UP_THRESHOLD = 0.7

# Define the gestures and their corresponding finger configurations
GESTURES = {
    'Fist': [0, 0, 0, 0, 0],
    'One': [1, 0, 0, 0, 0],
    'Two': [1, 1, 0, 0, 0],
    'Three': [1, 1, 1, 0, 0],
    'Four': [1, 1, 1, 1, 0],
    'Five': [1, 1, 1, 1, 1],
}

# Define the function for detecting the current gesture
def detect_gesture(hand_landmarks_list):
    finger_up_list = []
    for landmark_id in tip_landmark_ids:
        landmark = hand_landmarks_list.landmark[landmark_id]
        finger_up = landmark.y < FINGER_UP_THRESHOLD * image_height
        finger_up_list.append(finger_up)
    for gesture_name, gesture_finger_config in GESTURES.items():
        if finger_up_list == gesture_finger_config:
            return gesture_name
    return None

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic, \
        mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Mirror the image horizontally
        image = cv2.flip(image, 1)

        # Convert the image to RGB and process it with MediaPipe Holistic
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results_holistic = holistic.process(image_rgb)

        # Convert the image to RGB and process it with MediaPipe Hands
        image_hands = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, _, _ = image.shape
        results_hands = hands.process(image_hands)

        # Draw face landmarks on the image
        if results_holistic.face_landmarks:
            mp_drawing.draw_landmarks(
                image, results_holistic.face_landmarks, mp_holistic.FACEMESH_CONTOURS)

            # Mirror the face landmarks horizontally
            for landmark in results_holistic.face_landmarks.landmark:
                landmark.x = 1.0 - landmark.x

            # Get image height after processing
            image_height, _, _ = image.shape

            # Calculate average y-coordinate for each set of landmarks
            head_landmarks = [(landmark.x, landmark.y, landmark.z) for landmark in results_holistic.face_landmarks.landmark[:5]]
            left_eye_y = results_holistic.face_landmarks.landmark[159].y
            right_eye_y = results_holistic.face_landmarks.landmark[386].y
            mouth_landmarks_y = [landmark.y for landmark in results_holistic.face_landmarks.landmark[308:332]]

            # Calculate average values
            head_movement_y = map_to_midi(sum([coord[1] for coord in head_landmarks]) / len(head_landmarks), 0.0, 1.0)
            left_eye_movement_y = map_to_midi(left_eye_y, 0.0, 1.0)
            right_eye_movement_y = map_to_midi(right_eye_y, 0.0, 1.0)
            mouth_movement_y = map_to_midi(sum(mouth_landmarks_y) / len(mouth_landmarks_y), 0.0, 1.0)

            # Send OSC messages for face movements
            osc_message_face = [head_movement_y, left_eye_movement_y, right_eye_movement_y, mouth_movement_y]
            client.send_message("/face_movements", osc_message_face)

        # Draw hand landmarks on the image
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Mirror the hand landmarks horizontally
                for landmark in hand_landmarks.landmark:
                    landmark.x = 1.0 - landmark.x

                # Detect the current gesture and send OSC messages
                gesture_id = detect_gesture(hand_landmarks)
                if gesture_id:
                    client.send_message("/gesture", gesture_id)
                    for i, landmark_id in enumerate(tip_landmark_ids[:4]):
                        landmark = hand_landmarks.landmark[landmark_id]
                        x = int(landmark.x * 127)
                        y = int(landmark.y * 127)
                        z = int(landmark.z * 127)
                        client.send_message(f"/finger_{i}", [x, y, z])

        # Display the image
        cv2.imshow('MediaPipe Holistic and Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
