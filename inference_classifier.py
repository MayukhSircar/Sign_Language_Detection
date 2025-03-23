import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize camera
cap = cv2.VideoCapture(0)  # Adjust index if needed

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Labels for predictions
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4:'E', 5:'F',6:'G', 7:'H', 8:'I', 9:'J', 10:'K',11:'L',12:'M', 13:'N', 14:'O', 15: 'P', 16:'Q', 17:'R', 18:'S', 19:'T', 20:'U', 21:'V',22:'W', 23:'X', 24:'Y', 25:'Z'}  # Add more labels as per your model

# To accumulate letters and form words
current_word = []
display_word = ""
last_prediction = None  # To track the previous prediction
prediction_delay = 25  # Delay (in frames) between recognizing the same letter
delay_counter = 0       # Counter to implement delay

# Main loop
while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        print("Error: Camera not accessible.")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame for hand landmarks
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extract hand landmarks (x and y)
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Compute normalized x and y differences
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))  # Normalize x
                data_aux.append(y - min(y_))  # Normalize y

            # Ensure feature count matches model expectation
            if len(data_aux) == 42:
                # Predict the letter using the model with delay
                if delay_counter == 0:  # Only predict when delay is 0
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = labels_dict[int(prediction[0])]

                    if predicted_character != last_prediction:  # Avoid repetition
                        current_word.append(predicted_character)
                        last_prediction = predicted_character
                        delay_counter = prediction_delay  # Reset delay counter

                # Decrement delay counter
                if delay_counter > 0:
                    delay_counter -= 1

                # Draw prediction on the frame
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                cv2.rectangle(frame, (x1, y1), (x1 + 100, y1 - 40), (0, 0, 0), -1)
                cv2.putText(frame, last_prediction, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)

    # Display the current word
    display_word = ''.join(current_word)
    cv2.putText(frame, f"Word: {display_word}", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # Show the frame
    cv2.imshow('Sign Language Recognition', frame)

    # Controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit the program
        break
    elif key == ord('r'):  # Reset the word
        current_word = []
        display_word = ""
        last_prediction = None
        delay_counter = 0

# Release resources
cap.release()
cv2.destroyAllWindows()
