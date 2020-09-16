import cv2
# Our Image
img_file = "car5_img.jpg"
# video = cv2.VideoCapture('Tesla Autopilot Dashcam1.mp4')
video = cv2.VideoCapture('Tesla Autopilot Dashcam1.mp4')


# Our pre-trained car & pedestrian classifier
car_tracker = cv2.CascadeClassifier('car_detector.xml')
pedestrian_tracker = cv2.CascadeClassifier('haarcascade_fullbody.xml')


# Run forever until car stop
while True:
    # Read the current frame
    (read_successful, frame) = video.read()

    # Safe coding
    if read_successful:
        # Must convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # Detect cars & pedestrian
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    # Draw rectangle around the cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x+1, y+2), (x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Draw rectangle around the cars
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    # Display the image with face spotted
    cv2.imshow("Cars & Pedestrians Detector", frame)

    # Don't autoclose (Wait here in the codde and listen for a key press)
    key = cv2.waitKey(1)

    # Stop if Q key is pressed
    if key == 81 or key == 113:
        break

# Realese the VideoCapture object
video.release()

"""
# create opencv image
img = cv2.imread(img_file)

# convert it to grayscale (needed for haar cascade)
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# detect cars
cars = car_tracker.detectMultiScale(black_n_white)

# Draw rectangle around the cars
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

# Display the image with the face spotted
cv2.imshow("Car Detector", img)

# Don't autoclose (Wait here in the code and listen for a key press)
cv2.waitKey()

print("Code Completed")
"""
