# obtain_frames_cv2 : Program to obtain image frames from a video per second
# By Kshitij Pingle
# kpingle@csu.fullerton.edu
# CWID : 885626978
# 28 August 2024
# Modified : 30 July 2025

import cv2 
import os 


def create_frames(video : str, directory_name : str, fps = 60):
    """Create a folder of image frames for a mp4 video"""
    
    cam = cv2.VideoCapture(video)

    try: 	
        # creating a folder named data 
        if not os.path.exists(directory_name): 
            os.makedirs(directory_name) 

    # if not created then raise error 
    except OSError: 
        print ('Error: Creating directory of data') 

    # fps = int(cam.get(cv2.CAP_PROP_FPS))
    # frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    # seconds = round(frames / fps)

    # print("\nNumber of frames =", frames)
    # print("Frames per Second =", fps)
    # print("Length of video in seconds =", seconds)

    # Get name of video without the '.mp4'
    parts_of_string = video.split('.')
    # print("First part of the string:", parts_of_string[0])
    # print("Second part of the string:", parts_of_string[1])


    currentframe = 0
    count = 0

    while(True): 	
        # reading from frame 
        ret, frame = cam.read() 

        if ret: #If we got a frame

            if ((count % fps) == 0): #This ensures a frame every second
                name = directory_name + '/' + parts_of_string[0] + '_frame_' + str(currentframe) + '.jpg'
                
                if ((currentframe % 10) == 0): #Print message every 10 frames
                    print ('Creating...' + name) 

                # writing the extracted images 
                cv2.imwrite(name, frame)
                currentframe += 1

            # increasing counter to show
            count += 1
        else: 
            break

    #print("DEBUG: Count =", count)
    print("Finished creating frames\n")

    # Release all space and windows once done 
    cam.release() 
    cv2.destroyAllWindows()
#End of create_frames function


# Test
video = "video_09_17-41_February_10_2026.mp4"
directory_name = "video_09_17-41_February_10_2026_images"

create_frames(video, directory_name)
