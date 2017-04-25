"""
Slither Deep Learning Bot
By: James Caudill
For: CPE 461/462 Senior Project
Using: OpenCV, OpenAI Gym/Universe, TensorFlow
"""

import gym
import universe 
import numpy as np
import cv2
import imutils
from PIL import Image
import math

"""
Small translation helper functions (may use later)
"""
def translateSmallToBig(x,y):
  return x+20, y+86

def translateBigToSmall(x,y):
  return x-20, y-86


"""
Initialize the vision dictionary with Background (may use later)
"""
def initDict(visDict):
  # init dict
  for x in range(100):
    for y in range(60):
      visDict[(x, y)] = 'B'


"""
Desc: Take observations returned from env.step and return two imgs
Input: Observation array (basically just pixels)
Output:	img: labeling enviornment elements
	output: printig cv results on
"""
def processVision(ob):
  newOb = ob[0]['vision']		# yank the 3d numpy vision array
  img = newOb[86:386, 20:520, ::-1]	# img is what the cv operations will be on
  output = img[:,:,::-1]		# ouput is for printing results onto
 
  img = cv2.GaussianBlur(img,(3,3),0)	# blur for rounded cirlces

  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)			# grayscale
  ret, img = cv2.threshold(img, 60, 255, cv2.THRESH_TOZERO)	# threshold background
  
  cv2.circle(img, (435, 235), 45, (0,0,0), -1)			# mask map bot-right
  cv2.rectangle(img, (10, 260), (120, 300), (0,0,0), -1)	# mask score bot-left

  return img, output


"""
Desc: Take the image and update the vision dictionary
Input: 	img: the grayscaled image to find circles on
	output: the image to print results on
	visDict: the vision dictionary to mark mass on
Output: no return (edits by reference)
"""
def findMass(img, output, visDict):
  # find mass cirlces with hough circles
  circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 1,
                param1 = 10, param2 = 10, minRadius = 0, maxRadius = 5)
  if circles is not None:				# if mass found
    circles = np.uint16(np.around(circles))		# convert circles to np array
    for i in circles[0, :]:				# loop through circles
      visDict[(i[0]//5, i[1]//5)] = 'M'			# mark the mass in the vision dict
      cv2.circle(output, (i[0],i[1]), i[2], (0,255,0), 2)    	# mark mass with green circle
      # cv2.circle(img, (i[0],i[1]), i[2] + 1, (0,0,0), -1)    	# mask mass found
      # cv2.imshow('circles', output)			# show user the circles found


"""
Desc: Attempt to find out whether large object is a snake or mass
Input: The object (contour) in question
Output: Mass or Snake
"""
# def decideMass(contour):
  

"""
Desc: Determine your snake and other snakes borders
Input: 	img: the black and white image to find contours on
	output: the image to print results on
	visDict: the vision dictionary to mark snakes on
	centerX, centerY: the center of the image (to help find your snake)
Output: no return (edits by reference)
"""
def findSnakes(img, output, visDict, centerX, centerY):
  # find contours in the image (snakes and big overlapping mass)
  img, contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
  if contours is not None:		# if there are contours found
    close = 51				# sorta abritrary close value
    closestDist = 50			# definitely arbitrary
    closeCon = contours[0]		# just pick a close contour
    for con in contours:		# loop through all contours
      M = cv2.moments(con)		# take the moments for the contour
      area = cv2.contourArea(con)	# take the area of the contour

      if (area >= 170):			# smallest area for snek
        for point in con:		# loop through all returned points
          visDict[(point[0][0]//5, point[0][1]//5)] = 'S'	# mark as snake in vision dict

        cv2.drawContours(output, con, -1, (0,0,255), 2)		# show user the found contour
        if (M["m00"] != 0):			# check for divide by 0
          cX = int(M["m10"] / M["m00"])		# find centroid X
          cY = int(M["m01"] / M["m00"])		# find centroid Y
          close = math.hypot(cX - centerX, cY - centerY)	# find distance to center
        if (close < closestDist):		# check if its closer than previous
          closestDist = close			# update closest distance
          closeCon = con			# update closest contour
    cv2.drawContours(output, closeCon, -1, (255,0,0), 3)	# draw your snake
    for point in closeCon:					# loop through your snake points
      visDict[(point[0][0]//5, point[0][1]//5)] = 'Y'		# mark your snake in vision dict


def main():
  visionDict = {}	# create dictionary for vision input
  score = 10		# start the score at 10 (thats what the game does)
  centerX = 250		# init the center X (for small screen)
  centerY = 150		# init the center Y (for small screen)
  # make slither with no skins (also no top ten list)
  env = gym.make('internet.SlitherIONoSkins-v0')
  env.configure(remotes=1)  	# automatically creates a local docker container
  observation_n = env.reset()	# start the the game (pretty much)
  
  while True:						# loop 4ever
    if (observation_n[0] != None):			# if the game is live
      visionDict.clear()				# clear the dict for new objects
      img, output = processVision(observation_n)	# process the pixels
      cv2.imshow('gray', img)				# show user grayscaled

      findMass(img, output, visionDict)			# find the mass (dots)

      # make binary threshold for contours
      ret, img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
      cv2.imshow('b&w', img)				# show user black and white

      # find sneks
      findSnakes(img, output, visionDict, centerX, centerY)
      # print (visionDict)   

      cv2.imshow('all detected objects', output)	# show user everythang
      cv2.waitKey(0)					# WAIT
      cv2.destroyAllWindows()				# DESTROY			
    else:
      score = 10 					# reset score

    # TODO: decision making
    action_n = [[('PointerEvent', 50, 250, 0)] for ob in observation_n]  # your agent here
    observation_n, reward_n, done_n, info = env.step(action_n)
    if (reward_n[0] != 0):		# if your snake ate mass
      score = score + reward_n[0] 	# update score
      print (reward_n)			# print what you ate
      print (score)			# print how fat you are
    env.render()			# show what you got
  

if __name__ == "__main__":
	main()
