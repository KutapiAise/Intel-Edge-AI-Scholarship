from __future__ import absolute_import, division, print_function
import os
import sys
import time
from argparse import ArgumentParser
import pathlib
import cv2
import numpy as np
import json
from inference import Network
from collections import namedtuple

import dlib


from detect_drowsinessAPI import *

alarm ="resources/alarm.wav"
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30

MOUTH_OPEN_THRESH=20

is_async_mode = True
CONFIG_FILE = 'resources/config.json'
MULTIPLICATION_FACTOR = 5


MyStruct = namedtuple("drivingInfo", "driver, looker,msg")
INFO = MyStruct(0, 0,'')

# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0
ALARM_ON = False

yawns = 0
yawn_status = False 

#eye aspect ratio
ear=0 
# To get current working directory
CWD = os.getcwd()

# Creates subdirectory to save output snapshots
pathlib.Path(CWD + '/output_snapshots/').mkdir(parents=True, exist_ok=True)

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
#print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("resources/shape_predictor_68_face_landmarks.dat")

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]



def build_argparser():
    parser = ArgumentParser()

    parser.add_argument("-mf", "--modelface",
                        help="Path to an .xml file with a trained model for face detection.",
                        required=True, type=str)

    parser.add_argument("-mp", "--modelpose",
                        help="Path to an .xml file with a trained model for head pose.",
                        required=True, type=str)
     
    parser.add_argument("-l", "--cpu_extension",
                        help="MKLDNN (CPU)-targeted custom layers. Absolute "
                             "path to a shared library with the kernels impl.",
                        type=str, default=None)
   
    parser.add_argument("-d", "--device",
                        help="Specify the target device to infer on; "
                             "CPU, GPU, FPGA, HDDL or MYRIAD is acceptable. Application"
                             " will look for a suitable plugin for device "
                             "specified (CPU by default)", default="CPU", type=str)
    parser.add_argument("-pt", "--prob_threshold",
                        help="Probability threshold for detections filtering",
                        default=0.5, type=float)
    parser.add_argument("-f", "--flag", help="sync or async", default="async", type=str)

    return parser




def face_detection(res, initial_wh):
    """
    Parse Face detection output.

    :param res: Detection results
    :param initial_wh: Initial width and height of the FRAME
    :return: Co-ordinates of the detected face
    """
    global INFO
    faces = []
    INFO = INFO._replace(driver=0)

    for obj in res[0][0]:
        # Draw only objects when probability more than specified threshold
        if obj[2] > CONFIDENCE:
            if obj[3] < 0:
                obj[3] = -obj[3]
            if obj[4] < 0:
                obj[4] = -obj[4]
            xmin = int(obj[3] * initial_wh[0])
            ymin = int(obj[4] * initial_wh[1])
            xmax = int(obj[5] * initial_wh[0])
            ymax = int(obj[6] * initial_wh[1])
            faces.append([xmin, ymin, xmax, ymax])
            #("length of faces =: " ,len(faces))
            INFO = INFO._replace(driver=len(faces))
    return faces




def main():
    global CONFIG_FILE
    global is_async_mode
    global CONFIDENCE
    global POSE_CHECKED
    global INFO
    global COUNTER
    global ALARM_ON
    global yawns
    global yawn_status
    global EYE_AR_CONSEC_FRAMES
    global ear
    global leftEye
    global rightEye


        
    args = build_argparser().parse_args()

    try:
        CONFIDENCE = float(os.environ['CONFIDENCE'])
    except:
        CONFIDENCE = 0.5

    assert os.path.isfile(CONFIG_FILE), "{} file doesn't exist".format(CONFIG_FILE)
    config = json.loads(open(CONFIG_FILE).read())
    for idx, item in enumerate(config['inputs']):
        if item['video'].isdigit():
            input_stream = int(item['video'])
            cap = cv2.VideoCapture(input_stream)
            if not cap.isOpened():
                print("\nCamera not plugged in... Exiting...\n")
                sys.exit(0)
        else:
            input_stream = item['video']
            cap = cv2.VideoCapture(input_stream)
            if not cap.isOpened():
                print("\nUnable to open video file... Exiting...\n")
                sys.exit(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if args.flag == "async":
        is_async_mode = True
        print('Application running in async mode')
    else:
        is_async_mode = False
        print('Application running in sync mode')

    # Initialise the class
    infer_network = Network()
    
    infer_network_pose = Network()
    # Load the network to IE plugin to get shape of input layer
    plugin, (n_fd, c_fd, h_fd, w_fd) = infer_network.load_model(args.modelface, args.device, 1, 1, 2, args.cpu_extension)
    
    n_hp, c_hp, h_hp, w_hp = infer_network_pose.load_model(args.modelpose,args.device, 1, 3, 2,args.cpu_extension, plugin)[1]

    print("To stop the execution press Esc button")
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter(CWD + '/output_snapshots/outpy.mp4',0x00000021, 10, (initial_w,initial_h))

    frame_count = 1
   
    #ret, frame = cap.read()
    cur_request_id = 0
    next_request_id = 1

    while cap.isOpened():
        looking = 0
        ret, frame = cap.read()
        start_time = time.time()
        
        
        if not ret:
            break
        frame_count = frame_count + 1
        initial_wh = [cap.get(3), cap.get(4)]
        in_frame = cv2.resize(frame, (w_fd, h_fd))
        # Change data layout from HWC to CHW
        in_frame = in_frame.transpose((2, 0, 1))
        in_frame = in_frame.reshape( (n_fd, c_fd, h_fd, w_fd))

        # Start asynchronous inference for specified request.
        inf_start = time.time()
        if is_async_mode:
            infer_network.exec_net(next_request_id, in_frame)
        else:
            infer_network.exec_net(cur_request_id, in_frame)
        # Wait for the result
        if infer_network.wait(cur_request_id) == 0:
            det_time = time.time() - inf_start
            people_count = 0

            # Converting to Grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #Start region drowsiness detect
            # detect faces in the grayscale frame
            rects = detector(gray, 0)
            
            # loop over the face detections
            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                
                ear = (leftEAR + rightEAR) / 2.0
        
                leftEyeHull = cv2.convexHull(leftEye)
                #print(leftEyeHull, leftEyeHull.dtype)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                #to calculate yawn
                mouth = shape[mStart:mEnd]
                for (x, y) in   mouth:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                
                frame,lip_distance = mouth_open(frame)
                prev_yawn_status = yawn_status

                if ear < EYE_AR_THRESH :
                    COUNTER += 1
                    # if the eyes were closed for a sufficient number of times then sound the alarm
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        # if the alarm is not on, turn it on
                        if not ALARM_ON:
                            ALARM_ON = True
                            # check to see if an alarm file was supplied,
                            # and if so, start a thread to have the alarm
                            # sound played in the background
                            if alarm != "":
                                t = Thread(target=sound_alarm,	args=(alarm,))
                                t.deamon = True
                                t.start()
          

                else:
                    COUNTER =0
                    ALARM_ON = False

                if lip_distance > MOUTH_OPEN_THRESH:
                    yawn_status = True
                

                else:
                    yawn_status = False

                if prev_yawn_status == True and yawn_status == False:
                    yawns += 1
        
            #end region drowsiness
            
            
            # Results of the output layer of the network
            res = infer_network.get_output(cur_request_id)

           
            # Parse face detection output
            faces = face_detection(res, initial_wh)

            

            if len(faces) != 0:
                # Look for poses
                for res_hp in faces:
                    xmin, ymin, xmax, ymax = res_hp
                    head_pose = frame[ymin:ymax, xmin:xmax]
                    in_frame_hp = cv2.resize(head_pose, (w_hp, h_hp))
                    in_frame_hp = in_frame_hp.transpose((2, 0, 1))
                    in_frame_hp = in_frame_hp.reshape((n_hp, c_hp, h_hp, w_hp))

                    inf_start_hp = time.time()
                    infer_network_pose.exec_net(0, in_frame_hp)
                    infer_network_pose.wait(0)
                    det_time_hp = time.time() - inf_start_hp

                    # Parse head pose detection results
                    angle_p_fc = infer_network_pose.get_output(0, "angle_p_fc")
                    angle_y_fc = infer_network_pose.get_output(0, "angle_y_fc")
                    angle_r_fc = infer_network_pose.get_output(0, "angle_r_fc")
                    if ((angle_y_fc > -22.5) & (angle_y_fc < 22.5) & (angle_p_fc > -22.5) & (angle_p_fc < 22.5) & (angle_r_fc > -22.5) & (angle_r_fc < 22.5)):
                        looking += 1
                        POSE_CHECKED = True
                        INFO = INFO._replace(looker=looking)
                        #print("Subject is looking")
                        INFO =INFO._replace(msg="Looking Staright, you are doing great ! Keep it up!")
                    else:
                        INFO = INFO._replace(looker=looking)
                        #print("Subject is not looking")
                        INFO =INFO._replace(msg="WATCH THE ROAD!")
                        
                       
            else:
                INFO = INFO._replace(looker=0)
                

            time_interval = MULTIPLICATION_FACTOR * fps
            if frame_count % time_interval == 0:

                (frame, people_count)

        #frame = next_frame
        if is_async_mode:
            cur_request_id, next_request_id = next_request_id, cur_request_id
        #print("FPS : {}".format(1/(time.time() - start_time)))

              
        # Draw performance stats
        inf_time_message = "Face Inference time: N\A for async mode" if is_async_mode else \
            "Inference time: {:.3f} ms".format(det_time * 1000)

        
        head_inf_time_message = "Head pose Inference time: N\A for async mode" if is_async_mode else \
                "Inference time: {:.3f} ms".format(det_time_hp * 1000)
        cv2.putText(frame, head_inf_time_message, (0, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        log_message = "Async mode is on." if is_async_mode else \
            "Async mode is off."
        cv2.putText(frame, log_message, (0, 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
        cv2.putText(frame, inf_time_message, (0, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Driver: {}".format(INFO.driver), (0, 90), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 0, 0), 2)
 
        cv2.putText(frame, INFO.msg, (75, 90), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,255), 2)
        
        output_text = " Yawn frame Count: " + str(yawns)

        cv2.putText(frame, output_text, (0,110 ),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(230,0,0),2)

        if yawn_status == True:
            cv2.putText(frame, "Driver is Yawning!! BE AWAKE!!", (0,150), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),2)
        if COUNTER >= EYE_AR_CONSEC_FRAMES:
            cv2.putText(frame, "Drowsiness Alert!!", (400, 35),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, " Eye Aspect Ratio(EAR): {:.2f}".format(ear), (400, 55),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Driver is Awake!! " , (400, 35),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, "Eye Aspect Ratio(EAR): {:.2f}".format(ear) , (400, 55),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if ALARM_ON ==True:
            cv2.putText(frame, "BE AWAKE!! Alarm ON", (0,170), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),2)
        else:
            cv2.putText(frame, "Alarm OFF", (0,170), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),2)

        

        cv2.imshow("Detection Results", frame)
        # Write the frame into the file 'output.avi'
        out.write(frame)
        
        # Frames are read at an interval of 1 millisecond
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    infer_network.clean()
    infer_network_pose.clean()


if __name__ == '__main__':
    sys.exit(main() or 0)
