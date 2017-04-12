import gym
import universe  # register the universe environments
import numpy as np
import cv2
import imutils
from PIL import Image
import math

def processVision(ob):
  newOb = ob[0]['vision']
  img = newOb[86:386, 20:520, ::-1]
  output = img[:,:,::-1]
 
  #blur for rounded cirlces
  img = cv2.GaussianBlur(img,(5,5),0)

  # grayscale and threshold
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  ret, img = cv2.threshold(img, 60, 255, cv2.THRESH_TOZERO)
  # ret, img = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  
  # add map/score mask
  cv2.circle(output, (435, 235), 45, (0,0,0), -1)
  cv2.rectangle(img, (10, 260), (120, 300), (0,0,0), -1)

  return img, output

def findMass(img, output, visDict):
  # find mass cirlces
  circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 1,
                param1 = 10, param2 = 10, minRadius = 0, maxRadius = 5)
  if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
      visDict[(i[0]//5, i[1]//5)] = 'M'
      cv2.circle(output, (i[0],i[1]), i[2], (0,255,0), 2)    #mark mass with green circle
      cv2.circle(img, (i[0],i[1]), i[2] + 1, (0,0,0), -1)    #mask mass found
      # cv2.imshow('circles', output)


def findSnakes(img, output, visDict, centerX, centerY):
  # attempt to find your contour
  img, contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  # cv2.drawContours(output, contours, -1, (255,0,0), 3)
  if contours is not None:
    close = 51
    closestDist = 50
    closeCon = contours[0]
    for con in contours:
      M = cv2.moments(con)
      area = cv2.contourArea(con)
      # smallest area for snek
      if (area > 120):
        """
        print ("con")
        print (con)
        print ("con[0]")
        print (con[0])
        print ("con[0][0]")
        print (con[0][0])
        # x,y level
        print ("con[0][0][0]")
        print (con[0][0][0])
        """

        for point in con:
          visDict[(point[0][0]//5, point[0][1]//5)] = 'S'

        cv2.drawContours(output, con, -1, (0,0,255), 2)
        if (M["m00"] != 0):
          cX = int(M["m10"] / M["m00"])
          cY = int(M["m01"] / M["m00"])
          close = math.hypot(cX - centerX, cY - centerY)
        if (close < closestDist):
          closestDist = close
          closeCon = con
    cv2.drawContours(output, closeCon, -1, (255,0,0), 3)
    for point in closeCon:
      visDict[(point[0][0]//5, point[0][1]//5)] = 'Y'


def initDict(visDict):
  # init dict
  for x in range(100):
    for y in range(60):
      visDict[(x, y)] = 'B'

def main():

  visionDict = {}
  score = 10
  centerX = 250
  centerY = 150
  env = gym.make('internet.SlitherIONoSkins-v0')
  env.configure(remotes=1)  # automatically creates a local docker container
  observation_n = env.reset()
  # print ('Size of array: {}'.format(np.shape(observation_n[0])))
  
  # initDict(visionDict)
  # print (visionDict)

  while True:
    if (observation_n[0] != None):
      visionDict.clear()
      # initDict(visionDict)
      # print (type(observation_n))
      # print (type(observation_n[0]))
      # print (observation_n)
      # print (observation_n[0])
      img, output = processVision(observation_n)
      cv2.imshow('gray', img)

      findMass(img, output, visionDict)
      # print (visionDict)   

      # make binary threshold for contours
      ret, img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
      # cv2.imshow('b&w', img)

      findSnakes(img, output, visionDict, centerX, centerY)
      # print (visionDict)   

      cv2.imshow('all detected objects', output)
      cv2.waitKey(0)
      cv2.destroyAllWindows()

    # TODO: decision making
    action_n = [[('PointerEvent', 20, 385, 0)] for ob in observation_n]  # your agent here
    observation_n, reward_n, done_n, info = env.step(action_n)
    if (reward_n[0] != 0):
      score = score + reward_n[0] 
      print (reward_n)
      print (score)
    env.render()


if __name__ == "__main__":
	main()
