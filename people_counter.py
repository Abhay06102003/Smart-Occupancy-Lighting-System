from TRACK.CENTROID_TRACKER import CentroidTracker
from TRACK.TRACKABLE_OBJECT import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS 
from itertools import zip_longest
from Utils.mailer import Mailer
from Utils import thread
import numpy as np
import threading
import argparse
import datetime
import schedule
import logging
import imutils
import time
#import dlib
import json
import csv
import cv2

start_time = time.time()

logging.basicConfig(level=logging.INFO,format="[INFO] %(message)s")
logger = logging.getLogger(__name__)

with open("/home/abhay-06102003/Desktop/CROSS_COUNTING/Utils/config.json","r") as file:
    config = json.load(file)

print("Script started")

def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p","--prototxt",required=True,help= "path to Caffe 'deploy' prototxt file" )
    ap.add_argument("-m","--model",required=True,help= "path to Caffe pre-trained model")
    ap.add_argument("-i","--input",type = str,help = "path to optional input video file")
    ap.add_argument("-o","--output",type=str,help= "path to optional output file")
    ap.add_argument("-c","--confidence",type = float,default=0.4,help= "minimum confidence to filter weak decisions")
    ap.add_argument("-s","--skip-frames",type= int,default= 30,help= "# of skip frames between detections")
    args = vars(ap.parse_args())
    return args

def send_mail():
    Mailer().send(config["Email_Receive"])

def log_data(move_in,in_time,move_out,out_time):
    data = [move_in,in_time,move_out,out_time]
    export_data = zip_longest(*data,fillvalue='')
    
    with open("Utils/data/log/counting_data.csv","w",newline='') as myfile:
        wr = csv.writer(myfile,quoting = csv.QUOTE_ALL)
        if myfile.tell() == 0:
            wr.writerow("Move In","In Time","Move Out","Out Time")
            wr.writerows(export_data)
            
def people_counter():
    print("Starting function....")
    args = parse_arguments()
    print("Command-line arguments: ",args)
    CLASSES = [
        "background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"
        ]
    
    net = cv2.dnn.readNetFromCaffe(args["prototxt"],args["model"])
    
    if not args.get("input",False):
        logger.info("Starting the video..")
        vs = VideoStream(config["url"]).start()
        time.sleep(2.0)
        
    else:
        
        logger.info("Starting the Video..")
        vs = cv2.VideoCapture(args["input"])
    
    writer = None
    W = None
    H = None
    
    ct = CentroidTracker(maxDisappeared=40,maxDistance=50)
    trackers = []
    trackableobject = {}
    totalFrames = 0
    totalDown = 0
    totalUp= 0
    total = []
    move_out = []
    move_in = []
    out_time = []
    in_time = []
    
    fps = FPS().start()
    
    if config["Thread"]:
        vs = thread.ThreadingClass(config["url"])
    
    while True:
        frame = vs.read()
        frame = frame[1] if args.get("input",False) else frame
        
        if args["input"] is not None and frame is None:
            break
        
        frame = imutils.resize(frame,width = 500)
        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
        if W is None or H is None:
            (H,W) = frame.shape[:2]
        
        if args["output"] is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args["output"],fourcc,30,(W,H),True)
            
        status = "Waiting.."
        rects = []
        
        if totalFrames % args["skip_frames"] == 0:
            status = "Detecting.."
            trackers = []
            blob = cv2.dnn.blobFromImage(frame,0.007843,(W,H),127.5)
            net.setInput(blob)
            detections = net.forward()
            
            for i in np.arange(0,detections.shape[2]):
                confidence = detections[0,0,i,2]
                if confidence > args["confidence"]:
                    idx = int(detections[0,0,i,1])
                    if CLASSES[idx]!="person":
                        continue
                    
                    box = detections[0,0,i,3:7] * np.array([W,H,W,H])
                    (startX,startY,endX,endY) = box.astype("int")
                    tracker = cv2.TrackerCSRT_create()
                    bbox = (startX,startY,endX-startX,endY-startY)
                    tracker.init(rgb, bbox)
                    #tracker.setUseLearning(True)
                    #tracker = dlib.correlation_tracker()
                    #rect = dlib.rectangle(startX,startY,endX,endY)
                    #tracker.start_track(rgb,rect)
                    trackers.append(tracker)
        else:
            for tracker in trackers:
                status = "Tracking.."
                success,boxes = tracker.update(rgb)
                if success:
                    (startX,startY,w,h) = boxes
                    endX = startX+w
                    endY = startY+h
                
                #tracker.update(rgb)
                #pos = tracker.get_position()
                #startX = int(pos.left())
                #startY = int(pos.top())
                #endX = int(pos.right())
                #endY = int(pos.bottom())
                
                rects.append((startX,startY,endX,endY))
        
        cv2.line(frame,(0,H//2),(W,H//2),(0,0,0),3)
        cv2.putText(frame,"-Prediction border - Entrance-",(10,H-((i*20)+200)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
        objects = ct.update(rects)
        
        for (objectID,centroid) in objects.items():
            to = trackableobject.get(objectID,None)
            if to is None:
                to = TrackableObject(objectID, centroid)
            else:
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)
                
                if not to.counted:
                    if direction < 0 and centroid[1] < (H//2):
                        totalUp+=1
                        date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                        move_out.append(totalUp)
                        out_time.append(date_time)
                        to.counted = True
                    elif direction > 0 and centroid[1] > (H//2):
                        totalDown+=1
                        date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M" )
                        move_in.append(totalDown)
                        in_time.append(date_time)
                        if sum(total) >= config["Threshold"]:
                            cv2.putText(frame,"-ALERT: People Limit Exceeded-",(10,frame.shape[0]-80),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
                            if config["ALERT"]:
                                logger.info("Sending email alert..")
                                email_thread = threading.Thread(target = send_mail)
                                email_thread.daemon = True
                                email_thread.start()
                                logger.info("Alert sent!")
                        to.counted = True
                        total = []
                        total.append(len(move_in) - len(move_out))
            trackableobject[objectID] = to
            text = "ID {}".format(objectID)
            cv2.putText(frame,text,(centroid[0]-10,centroid[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
            cv2.circle(frame,(centroid[0],centroid[1]),4,(255,255,255),-1)
        
        info_status = [
            ("Exit",totalUp),
            ("Enter",totalDown),
            ("Status",status),
            ]
        info_total = [("Total People inside",', '.join(map(str,total))),]
        for (i,(k,v)) in enumerate(info_status):
            text = "{} {}".format(k,v)
            cv2.putText(frame,text,(10,H-((i*20)+20)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)
        for (i,(k,v)) in enumerate(info_total):
            text = "{} {}".format(k,v)
            cv2.putText(frame,text,(265,H-((i*20)+60)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        
        if config["Log"]:
            log_data(move_in,in_time,move_out,out_time)
            
        if writer is not None:
            writer.write(frame)
        
        cv2.imshow("Real-Time Monitoring",frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        
        totalFrames+=1
        fps.update()
        
        if config["Timer"]:
            end_time = time.time()
            num_seconds = (end_time-start_time)
            if num_seconds > 28800:
                break
    
    fps.stop()
    logger.info("Elapsed time: {:.2f}".format(fps.elapsed()))
    logger.info("Approx. FPS: {:.2f}".format(fps.fps()))
    
    if config["Thread"]:
        vs.release()
    
    cv2.destroyAllWindows()
        


if config["Scheduler"]:
    schedule.every().day.at("00:00").do(people_counter)
    while True:
        schedule.run_pending()
else:
    people_counter()
        
                                
                    