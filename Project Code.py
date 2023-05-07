import cv2
import mediapipe as mp
import numpy as np
import csv
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#taking personal data
with open ('C:/Users/sanja/OneDrive/Desktop/major project/weight6.csv','a+',newline='') as file:
    myfile = csv.writer(file)
    stdName = input("Enter you name : ")
    WCatagory = input("Enter workout Catagories(Internediate/ Advanced/ Beginner) : ")
    pushUp = int(input("Enter no of Push-ups in average : "))
    wPlace = input("Enter place you  workout usually( Gym/ Outdoor/ Home) : ")
    bWeight= int(input("Enter your weight in kg : "))
    Height =  float(input("Enter your height in ft : "))
    times = int(input("Enter no of days you do workout in a week : "))
    wBicepsset = int(input("Enter no of workout sets you perform : "))
    wRepetitioninset = int(input("Enter no of repeatation you perform in a set : "))
    myfile.writerow([stdName,WCatagory,pushUp,wPlace,bWeight,Height,times,wBicepsset,wRepetitioninset])

#data reading
df = pd.read_csv('C:/Users/sanja/OneDrive/Desktop/major project/weight6.csv')
dfs = df.astype(str)

#preprocessing and fitting
dfs['Tags'] =  dfs['Workout Categories']+" " +dfs['Avg']+" " +dfs['Place']+" " +dfs['bWeight']+" " +dfs['Hgt']+" " +dfs['Sets']+" " +dfs['Rpt']+" " +dfs['Times']
dfn = dfs[['Name','Tags','dWeight','cAng', 'rAng']]

#vectorizations
cv = CountVectorizer(max_features = 1000, stop_words= 'english')
vectors = cv.fit_transform(dfn['Tags']).toarray()

#cosine similarity
similarity = cosine_similarity(vectors)


# Recomendation System
def recommend(naam):
    name_index = dfn[dfn['Name']== naam].index[0]    #fetching indexs of name
    distances = similarity[name_index]# finding distances
    #for sorting
    # #enumerate include the index part
    # #key helps me to sort on respective column
    weight = sorted(list(enumerate(distances)),reverse = True,key = lambda x:x[1])[1:2]
    global mini
    global maxi
    #to select first data and then printing the recommendations
    ind_wt = weight[0][0]

    print("-------------------------------------")
    print("The Recommendation system recommends")

    print('your suggested weight(kg) is : ', dfn.iloc[ind_wt].dWeight)
    cang = dfn.iloc[ind_wt].cAng
    rang = dfn.iloc[ind_wt].rAng


    print('contraction angles : ',cang)
    print('relaxation angles : ',rang)
    cng = float(cang)
    rng = float(rang)
    mini = int(cng)
    maxi = int(rng)


#system call
recommend(stdName)

# Angle
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


cap = cv2.VideoCapture(0)

# Curl counter variables
counter = 0
stage = None

## Setup mediapipe instance
# for determination being level of accuracy in detection & tracking and providing good efficiency used 0.5
# stored all these in a single variable called pose to act as one
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)

            # Visualize angle
            cv2.putText(image, str(angle),
                        tuple(np.multiply(elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
            cv2.putText(image, str(angle),
                        tuple(np.multiply(elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )

            # Curl counter logic & angle
            if angle > maxi:
                stage = "down"
            if angle < mini and stage == 'down':
                stage = "up"
                counter += 1
                print(counter)

        except:
            pass

        # Render curl counter
        # Setup status box
        cv2.rectangle(image, (0, 0), (250, 73), (245, 117, 16), -1)  # data of rectangular box

        # Rep data & positions
        cv2.putText(image, 'REPS', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Stage data & positions to show
        cv2.putText(image, 'STAGE', (95, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, (95, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=3),
                                  # for nodes
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  # for edges
                                  )

        cv2.imshow('Mediapipe Feed of User Video', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# mp_drawing.DrawingSpec?? //foqr drawing whole body


len(landmarks)


landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility
landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]

landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

shoulder, elbow, wrist
calculate_angle(shoulder, elbow, wrist)
tuple(np.multiply(elbow, [640, 480]).astype(int))