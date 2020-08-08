import time #set intervals (seconds)
import cv2 #opencv import for image detection
import numpy as np #math and array stuff
import math



#code to capture video of classroom
# cap = cv2.VideoCapture(0)

# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret==True:
        

#         # write the flipped frame
#         out.write(frame)

#         cv2.imshow('Social distancing analyser',frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break
# cap.release()
# out.release()
# cv2.destroyAllWindows()

confid = 0.5 
thresh = 0.5


#change to video file in "videos" if using a pre-recorded video
vid_path = "./videos.Recording.mp4"  #we need to feed live footage into this. 

# Calibration needed for each video

def distance(p1, p2):
   return (math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))
        
def prox(p1, p2):
    cd = distance(p1, p2)
    if 0 < cd < 75.0:   #social distancing threshhold is approximately 6ft = 75
        return 1 
    elif 0 < cd < 100:  #change this valye
        return 2
    else:
        return 0

labelsPath = "./coco.names" #object detection
LABELS = open(labelsPath).read().strip().split("\n")
np.random.seed(42) #pseudo-random number that makes the code repeatable, and keep generating random inputs. Without this, there will be different outputs

#yolo conducts a single network pass and detects objects and locations
#we can try identifying where people in the room are not wearing masks
weightsPath = "./yolov3.weights" #connects to coco
configPath = "./yolov3.cfg" # configuration file "specifies the metadata needed to run the model, like the list of class names and where to store weights, as well as what data to use for evaluation"

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath) #darknet is a neural framework network
ln = net.getLayerNames() #get layer names
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()] #get indexes of output layers (this is what shows up)

vs = cv2.VideoCapture(vid_path) #sends vid to opencv
writer = None
(W, H) = (None, None)


q = 0
while True:
    (grabbed, frame) = vs.read() #returns bytes from file
    if not grabbed:
        break
    if W is None or H is None:
        (H, W) = frame.shape[:2] 
        q = W

    frame = frame[0:H, 200:q]
    (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False) #preps image for clasification, no cropping
   
 # construct a blob from the input image and then perform a forward
# pass of the YOLO object detector, giving us our bounding boxes and
# associated probabilities
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

# initialize our lists of detected bounding boxes, confidences, and
# class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

##looping through each layer
    for output in layerOutputs:
    ##looping through each detection
        for detection in output:
  #extract information
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

    #only do function on person
            if LABELS[classID] == "person": 
     #confidence should be above 50%
                if confidence > confid:
    #creating the boxes
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
   #updating the information
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
#prevents multiple boxes from showing up (gets rid of the weak ones)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confid, thresh)
#ensures at least one box exists
    if len(idxs) > 0:
  
  
        status = list()
        idf = idxs.flatten()
        close_pair = list()
        s_close_pair = list()
        center = list()
        dist = list()

       #get the cordinates of the boxes 
        for i in idf:
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            center.append([int(x + w / 2), int(y + h / 2)])
            
            status.append(0)
        for i in range(len(center)):
            for j in range(len(center)):
                g = prox(center[i], center[j])

                if g == 1:

                    close_pair.append([center[i], center[j]])
                    status[i] = 1
                    status[j] = 1
                elif g == 2:
                    s_close_pair.append([center[i], center[j]])
                    if status[i] != 1:
                        status[i] = 2
                    if status[j] != 1:
                        status[j] = 2
      #tally each one
        tot = len(center)
        low = status.count(2)
        high = status.count(1)
        safe = status.count(0)
        value = 0

        for i in idf:

            sub_img = frame[10:70, 10:W - 10]
            black_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0
            res = cv2.addWeighted(sub_img, 0.77, black_rect, 0.23, 1.0)

            frame[10:70, 10:W - 10] = res

     #create report part
            tot_str = "TOTAL STUDENTS: " + str(tot)
            high_str = "STUDENTS AT HIGH RISK: " + str(high)
            low_str = "STUDENTS AT LOW RISK: " + str(low)
            safe_str = "STUDENTS THAT ARE SAFE: " + str(safe)
            percent = "% AT HIGH RISK: " +str((high/tot)*100) + "%"
   
# Blue color in BGR 
            

            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            message = 0
            xcount = 0
            ycount = 0      
            count = 0
     #create different color rectangle accordingly
            if status[value] == 1:
                count = count + 1
                message = 1
                xcount = ((w)/2)+ xcount
                ycount = (h/2)+ ycount
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 150), 2)
                cv2.putText(frame, "THERE ARE STUDENTS IN DANGER", (160, H - 340),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, "VIEW PROBLEMATIC AREA BELOW", (165, H - 330),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            elif status[value] == 0:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 120, 255), 2)

            value += 1
        for h in close_pair:
            cv2.line(frame, tuple(h[0]), tuple(h[1]), (0, 0, 255), 2)
        for b in s_close_pair:
            cv2.line(frame, tuple(b[0]), tuple(b[1]), (0, 255, 255), 2)
       
        cv2.putText(frame, percent, (10, H - 370),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 200), 1)
        
        cv2.putText(frame, tot_str, (10, H - 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(frame, safe_str, (10, H - 350),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        cv2.putText(frame, low_str, (10, H - 340),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 120, 255), 1)
        cv2.putText(frame, high_str, (10, H - 330),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 200), 1)
    if message == 1:
        cv2.putText(frame, "problematic area", (int(xcount/count), H - int(ycount/count)),
                        cv2. FONT_HERSHEY_SIMPLEX , 0.9, (0, 0, 300), 3)
    cv2.imshow('Social distancing analyser', frame)
       
    cv2.waitKey(1)


#how you close out program 
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter("output.mp4", fourcc, 30,
                                 (frame.shape[1], frame.shape[0]), True)

    writer.write(frame)
print("Processing finished: open output.mp4")
writer.release()
writer.out.release()
vs.release()
