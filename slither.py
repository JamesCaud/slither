"""
Slither Deep Learning Bot
By: James Caudill
For: CPE 461/462 Senior Project
Using: OpenCV, OpenAI Gym/Universe, TensorFlow
"""

import gym
from gym import wrappers
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
  img2 = img				# img2 for dead mass cv operations
  output = img[:,:,::-1]		# ouput is for printing results onto
 
  img = cv2.GaussianBlur(img,(5,5),0)	# blur for rounded cirlces
  img2 = cv2.GaussianBlur(img2,(5,5),0)	# blur for rounded cirlces

  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)			# grayscale
  ret, img = cv2.threshold(img, 60, 255, cv2.THRESH_TOZERO)	# threshold background
  img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)			# grayscale
  ret, img2 = cv2.threshold(img2, 180, 255, cv2.THRESH_TOZERO)	# high white threshold 
  
  cv2.circle(img, (435, 235), 45, (0,0,0), -1)			# mask map bot-right
  cv2.rectangle(img, (10, 260), (120, 300), (0,0,0), -1)	# mask score bot-left
  cv2.circle(img2, (435, 235), 45, (0,0,0), -1)			# mask map bot-right
  cv2.rectangle(img2, (10, 260), (120, 300), (0,0,0), -1)	# mask score bot-left

  return img, img2, output


"""
Desc: Take the image and update the vision dictionary
Input: 	img: the grayscaled image to find circles on
	output: the image to print results on
	visDict: the vision dictionary to mark mass on
Output: point of biggest mass
"""
def findMass(img, output, visDict):
  bigMass = 1.0
  returnPoint = None
  # find mass cirlces with hough circles
  circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 1,
                param1 = 10, param2 = 10, minRadius = 0, maxRadius = 5)
  if circles is not None:				# if mass found
    circlesSorted = sorted(circles[0,:], key=lambda x: (x[2], 
                           math.hypot(x[0]-250, x[1]-150)), reverse=True)
    circle = circlesSorted[0]
    returnPoint = (circle[0], circle[1])
    visDict[(circle[0]//5, circle[1]//5)] = 'M'			# mark the mass in the vision dict
    cv2.circle(output, (circle[0], circle[1]), circle[2], (0,255,0), 2)    	# mark mass with green circle

  """
    print (circlesSorted)
    print ('\n')
    for i in circles[0, :]:				# loop through circles 
      if (i[2] > bigMass):
        bigMass = i[2]
        returnPoint = (i[0], i[1])
      visDict[(i[0]//5, i[1]//5)] = 'M'			# mark the mass in the vision dict
      cv2.circle(output, (i[0],i[1]), i[2], (0,255,0), 2)    	# mark mass with green circle
      # cv2.circle(img, (i[0],i[1]), i[2] + 1, (0,0,0), -1)    	# mask mass found
      # cv2.imshow('circles', output)			# show user the circles found
  """
  
  return returnPoint



"""
Desc: Determine if there is any dead snake mass on screen
Input: 	img: the black and white image to find contours on
	output: the image to print results on
	visDict: the vision dictionary to mark the mass on
		(this will be overwritting the the findSnakes mark of snake from find snakes)
Output: point of biggest mass
"""
def findDeadMass(img, output, visDict):
  bigMass = 0
  returnPoint = None
  # find the mass left behind by a dead snake
  img, contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  if contours is not None:			# if there are contours found
    for con in contours:			# loop through the contours
      M = cv2.moments(con)
      area = cv2.contourArea(con)		# calculate area of contour
      if (area > 130):				# (try to factor out snake heads)
        if (area > bigMass and M["m00"] != 0):
          bigMass = area
          returnPoint = (int(M["m10"]/M["m00"]), int(M["m01"] / M["m00"]))
        cv2.drawContours(output, con, -1, (0,255,0), 3)		# draw the contour on the output
        for point in con: 					# loop through points
          visDict[(point[0][0]//5, point[0][1]//5)] = 'M'	# mark as mass in vision dict
  
  return returnPoint


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
  img, contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  if contours is not None:		# if there are contours found
    close = 51				# sorta abritrary close value
    closestDist = 50			# definitely arbitrary
    closeCon = contours[0]		# just pick a close contour
    for con in contours:		# loop through all contours
      M = cv2.moments(con)		# take the moments for the contour
      area = cv2.contourArea(con)	# take the area of the contour

      if (area >= 170):			# smallest area for snek
        cv2.drawContours(output, con, -1, (0,0,255), 2)		# show user the found contour
        for point in con:		# loop through all returned points
          visDict[(point[0][0]//5, point[0][1]//5)] = 'S'	# mark as snake in vision dict

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
  mouse = (250, 150)
  mouseClick = 0
  score = 10		# start the score at 10 (thats what the game does)
  centerX = 250		# init the center X (for small screen)
  centerY = 150		# init the center Y (for small screen)
  # make slither with no skins (also no top ten list)
  env = gym.make('internet.SlitherIONoSkins-v0')
  # env = wrappers.Monitor(env, 'tmp/slither-test-1')
  env.configure(remotes=1)  	# automatically creates a local docker container
  observation_n = env.reset()	# start the the game (pretty much)
  
  while True:						# loop 4ever
    if (observation_n[0] != None):			# if the game is live
      visionDict.clear()				# clear the dict for new objects
      img, img2, output = processVision(observation_n)	# process the pixels
      # cv2.imshow('gray', img)				# show user grayscaled
      # cv2.imshow('higher threshold', img2)		# show user high threshold

      pointer = findMass(img, output, visionDict)	# find the mass (dots)
      if (pointer != None):
        mouse = pointer

      # make binary threshold for contours
      ret, img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
      ret, img2 = cv2.threshold(img2, 5, 255, cv2.THRESH_BINARY)
      # cv2.imshow('b&w', img)				# show user black and white
      # cv2.imshow('b&w2', img2)				# show user black and white

      # find sneks
      findSnakes(img, output, visionDict, centerX, centerY)
      # print (visionDict)   

      pointer = findDeadMass(img2, output, visionDict)# find dead mass on screen
      if (pointer != None):
        mouse = pointer
        mouseClick = 1
      else:
        mouseClick = 0
        pass

      # cv2.imshow('all detected objects', output)	# show user everythang
      # cv2.waitKey(0)					# WAIT
      # cv2.destroyAllWindows()				# DESTROY			
    else:
      score = 10 					# reset score

    newX, newY = translateSmallToBig(mouse[0], mouse[1])


    # TODO: decision making
    action_n = [[('PointerEvent', newX, newY, mouseClick)] for ob in observation_n]
    observation_n, reward_n, done_n, info = env.step(action_n)
    # if (done_n[0]):
    #   env.close()
    #   return
    if (reward_n[0] != 0):		# if your snake ate mass
      score = score + reward_n[0] 	# update score
      print (reward_n)			# print what you ate
      print (score)			# print how fat you are
    env.render()			# show what you got
  

if __name__ == "__main__":
	main()
