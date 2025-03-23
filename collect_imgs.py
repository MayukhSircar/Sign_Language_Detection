import cv2
import os

# Directory to store the collected images
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 26
dataset_size = 100

# Try to open the camera (change 2 to 0 or 1 if needed)
cap = cv2.VideoCapture(0)  # Try other indices like 0 or 1 if 2 doesn't work
if not cap.isOpened():
    print("Error: Camera not accessible")
    exit()

# Loop through each class
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print(f'Collecting data for class {j}')

    done = False
    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            print("Error: Unable to read frame.")
            break

        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Collecting and saving images
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            print("Error: Unable to read frame.")
            break

        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), f'{counter}.jpg'), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()
