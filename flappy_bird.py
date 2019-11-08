import pygame
from pygame.locals import *  # noqa
import sys
import random
import numpy as np
from collections import deque

#np.random.seed(1)

class AI:
    def __init__(self):
        self.data = list()
        self.answers = list()

        self.hidden1_size = 160
        self.hidden2_size = 410
        self.hidden3_size = 12
        self.input_size = 40

        self.maxScore = 0
        self.createNewAi()

        self.activation_func = self.tanh
        self.activation_func_deriv = self.tanh2deriv
        self.output_func = lambda x: x
        self.error_func = self.avg_square_error
        
        self.weights_01_prev = 2 * np.random.random((self.input_size, self.hidden1_size)) - 1
        self.weights_12_prev = 2 * np.random.random((self.hidden1_size,self.hidden2_size)) - 1
        self.weights_23_prev = 2 * np.random.random((self.hidden2_size,self.hidden3_size)) - 1
        self.weights_34_prev = 2 * np.random.random((self.hidden3_size,1)) - 1

    def createNewAi(self):
        
        self.learn_rate = 0.00005

        if(self.maxScore > 2):
            print("restoring successful AI")
            self.weights_01 = self.weights_01_prev
            self.weights_12 = self.weights_12_prev
            self.weights_23 = self.weights_23_prev
            self.weights_34 = self.weights_34_prev
        else:
            print("creating new AI")
            self.weights_01 = 2 * np.random.random((self.input_size, self.hidden1_size)) - 1
            self.weights_12 = 2 * np.random.random((self.hidden1_size,self.hidden2_size)) - 1
            self.weights_23 = 2 * np.random.random((self.hidden2_size,self.hidden3_size)) - 1
            self.weights_34 = 2 * np.random.random((self.hidden3_size,1)) - 1

        self.maxScore = 0

        

        
    def addGameData(self, new_data):
        self.data.append(new_data)
        
        #sys.stdout.write("                                           \r"+ str(new_data))
        #self.printAiInfo()

        prog = self.makeProg(new_data)
        self.answers.append(prog)
        return prog
    
    def makeProg(self, data):
        layer_0 = np.array(data)
        layers = self.calcLayers(layer_0)
        return layers[-1]

    def calcLayers(self, layer_0):
        layer_1 = self.activation_func(np.dot(layer_0, self.weights_01))
        layer_2 = self.activation_func(np.dot(layer_1, self.weights_12))
        layer_3 = self.activation_func(np.dot(layer_2, self.weights_23))
        layer_4 = self.output_func(np.dot(layer_3, self.weights_34))
        return (layer_1,layer_2,layer_3,layer_4)
    
    def printAiInfo(self):
        return
        #print(len(self.data))

    def iterateCycle(self, is_dead, score):
        self.trainBatch(not is_dead, score)
        self.data.clear()
        self.answers.clear()
        #print("dead" if is_dead else "alive")

    def trainBatch(self, is_positive, score):
        layer_output_error = 0
        
        batch_len = len(self.answers)
        batch_len = batch_len if batch_len > 0 else 1
        magic = np.log((batch_len/13)**2)
        magic = magic if magic > 0 else 1
        score = 1 + score / 100
        endorse_power = magic * score if is_positive else score
        
        #print(batch_len)
        #print(error_power)
        #print(endorse_power)
        for i in range(10 if is_positive else 0, batch_len): #10 first are stubbed data
            
            layer_0 = np.array(self.data[i:i+1])
            layers =  self.calcLayers(layer_0)
            layer_1 = layers[0]

            if (not is_positive):
                dropout_mask = np.random.randint(2,size=layer_1.shape)
                layer_1 *= dropout_mask * 2

            layer_2 = layers[1]
            if (not is_positive):
                dropout_mask = np.random.randint(2,size=layer_2.shape)
                layer_2 *= dropout_mask * 2

            layer_3 = layers[2]
            if (not is_positive):
                dropout_mask = np.random.randint(2,size=layer_3.shape)
                layer_3 *= dropout_mask * 2

            layer_4 = layers[3]

            goal = self.answers[i:i+1][0]
            goal = goal if is_positive else (-goal)

            layer_output_error += self.error_func(layer_4, [goal])
            
            #производная от среднеквадратического отклонения - скаляр
            layer_4_delta = (layer_4 - goal)
            
            #back propagation func:
            #поэлементное произведение весов последнего скрытого слоя (рез. - вектор длины скрытого слоя)
            #умноженное на производную от функции активации
            layer_3_delta = layer_4_delta * self.weights_34.T * self.activation_func_deriv(layer_3)
            layer_2_delta = layer_3_delta.dot(self.weights_23.T) * self.activation_func_deriv(layer_2)
            layer_1_delta = layer_2_delta.dot(self.weights_12.T) * self.activation_func_deriv(layer_1)

            if (score > self.maxScore):
                self.weights_01_prev = self.weights_01
                self.weights_12_prev = self.weights_12
                self.weights_23_prev = self.weights_23
                self.weights_34_prev = self.weights_34
                self.maxScore = score

            amp =  self.learn_rate * endorse_power
            if (not is_positive):
                amp *= 0.6      
            if (not (not is_positive and ((i % 10) != 0))):
                self.weights_34 -= amp * layer_3.T.dot(layer_4_delta)
                self.weights_23 -= amp * layer_2.T.dot(layer_3_delta)
                self.weights_12 -= amp * layer_1.T.dot(layer_2_delta)
                self.weights_01 -= amp * layer_0.T.dot(layer_1_delta)

            

                

        if (is_positive):
            self.learn_rate -= self.learn_rate / 1000
        else:
            self.learn_rate += self.learn_rate / 10000

        #print(self.answers)
        #print("error: " + str(layer_3_error))
    
    def tanh(self,x):
        return np.tanh(x)

    def tanh2deriv(self,output):
        return 1 - (output ** 2)

    def softmax(self,x):
        temp = np.exp(x)
        return temp / np.sum(temp, axis=0, keepdims=True)

    def avg_square_error(self,np_arr, goal):
        return np.sum((np_arr - goal) ** 2)


class FlappyBird:
    def __init__(self):
        self.screen = pygame.display.set_mode((400, 708))
        self.bird = pygame.Rect(65, 50, 50, 50)
        self.background = pygame.image.load("assets/background.png").convert()
        self.birdSprites = [pygame.image.load("assets/1.png").convert_alpha(),
                            pygame.image.load("assets/2.png").convert_alpha(),
                            pygame.image.load("assets/dead.png")]
        self.wallUp = pygame.image.load("assets/bottom.png").convert_alpha()
        self.wallDown = pygame.image.load("assets/top.png").convert_alpha()
        self.gap = 200  #300 ez #130 hard
        self.wallx = 400
        self.birdY = 350
        self.jump = 0
        self.jumpSpeed = 10
        self.gravity = 5
        self.dead = False
        self.sprite = 0
        self.counter = 0
        self.offset = random.randint(-110, 110)

        self.ai = AI()
        self.frame = 0
        self.iteration = 0
        initGI = self.getGameInfoForAi()
        self.prevGameInfo = deque([initGI,initGI,initGI,initGI,initGI,initGI,initGI,initGI,initGI])
        self.lastAiCommand = 0.
        self.framerate = 10000
        self.maxScore = 0

    def updateWalls(self):
        self.wallx -= 2
        if self.wallx < -80:
            self.wallx = 400
            self.offset = random.randint(-110, 110)

            if not self.dead:
                self.counter += 1
                if self.counter%20 == 0 and self.gap > 130:
                    self.gap -= 10
                self.maxScore = np.max([self.counter, self.maxScore])
                self.iterateAiCycle(False)

    def birdUpdate(self):
        if self.jump:
            self.jumpSpeed -= 1
            self.birdY -= self.jumpSpeed
            self.jump -= 1
        else:
            self.birdY += self.gravity
            self.gravity += 0.2
        self.bird[1] = self.birdY
        
        upRect = pygame.Rect(self.wallx,
                             360 + self.gap - self.offset + 10,
                             self.wallUp.get_width() - 10,
                             self.wallUp.get_height())
        downRect = pygame.Rect(self.wallx,
                               0 - self.gap - self.offset - 10,
                               self.wallDown.get_width() - 10,
                               self.wallDown.get_height())
        if upRect.colliderect(self.bird) or downRect.colliderect(self.bird):
            self.dead = True
        
        if not 0 < self.bird[1] < 720:
            self.bird[1] = 50
            self.birdY = 350
            self.dead = False
            if self.counter > 0:
                print("score: " + str(self.counter))
            self.counter = 0
            self.wallx = 400
            self.offset = random.randint(-110, 110)
            self.gravity = 5

            self.iterateAiCycle(True)

    def iterateAiCycle(self, is_dead):
        self.iteration += 1
        if (self.maxScore < 30):
            self.ai.iterateCycle(is_dead, self.counter)
        every = 500
        if (self.iteration % every == 0) and (self.maxScore < (self.iteration / every)) and self.maxScore < 30:
            self.ai.createNewAi()
            self.iteration = 0
            self.maxScore = 0
        

    def updateAi(self):
        self.frame += 1
        if (self.frame >= self.framerate):
            self.frame = 0
        
        if((self.frame % 1) == 0 and not self.dead):
            new_info = self.getGameInfoForAi()
            lst = list(self.prevGameInfo)
            gameInfo = list(np.array(lst).flatten()) + list(new_info)
            self.prevGameInfo.rotate()
            self.prevGameInfo.popleft()
            self.prevGameInfo.append(new_info)

            self.lastAiCommand = self.ai.addGameData(gameInfo)
            

    def getGameInfoForAi(self):
        return [self.birdY / 600, (self.wallx + 200) / 1000,  (self.offset + 100) / 200, (self.offset + self.gap + 100) / 200]

    def run(self):
        clock = pygame.time.Clock()
        pygame.font.init()
        font = pygame.font.SysFont("Arial", 50)
        while True:
            clock.tick(self.framerate)

            self.updateAi()

            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
            """
                if (event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN) and not self.dead:
                    self.jump = 17
                    self.gravity = 5
                    self.jumpSpeed = 10

            """
            
            if (event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN):
                if(self.framerate < 1000):
                    self.framerate = 1000
                else:
                    self.framerate = 60

            if(self.lastAiCommand > 0 and not self.dead):
                self.jump = 17
                self.gravity = 5
                self.jumpSpeed = 10
            

            self.screen.fill((255, 255, 255))
            self.screen.blit(self.background, (0, 0))
            self.screen.blit(self.wallUp,
                             (self.wallx, 360 + self.gap - self.offset))
            self.screen.blit(self.wallDown,
                             (self.wallx, 0 - self.gap - self.offset))
            self.screen.blit(font.render(str(self.counter),
                                         -1,
                                         (255, 255, 255)),
                             (200, 50))
            if self.dead:
                self.sprite = 2
            elif self.jump:
                self.sprite = 1
            self.screen.blit(self.birdSprites[self.sprite], (70, self.birdY))
            if not self.dead:
                self.sprite = 0
            self.updateWalls()
            self.birdUpdate()
            pygame.display.update()

if __name__ == "__main__":
    FlappyBird().run()