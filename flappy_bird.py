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

        self.hidden1_size = 60
        self.hidden2_size = 128
        self.hidden3_size = 7
        self.input_size = 40

        self.createNewAi()

        self.activation_func = self.tanh
        self.activation_func_deriv = self.tanh2deriv
        self.output_func = lambda x: x
        self.error_func = self.avg_square_error

        self.epochs_count = 20
        self.epoch_size = 5000
        self.successful_batches = deque(maxlen=self.epoch_size)
        

    def createNewAi(self):
        self.learn_rate = 0.00001
        self.avg_error = 0
        self.max_score = 0

        self.teach_tries = 1

        self.weights_01 = 0.2 * np.random.random((self.input_size, self.hidden1_size)) - 0.1
        self.weights_12 = 0.5 * np.random.random((self.hidden1_size,self.hidden2_size)) - 0.25
        self.weights_23 = 0.5 * np.random.random((self.hidden2_size,self.hidden3_size)) - 0.25
        self.weights_34 = 1 * np.random.random((self.hidden3_size,1)) - 1
        print()
        print("creating new AI")

        
    def addGameData(self, new_data):
        self.data.append(new_data)
        
        #sys.stdout.write("                                           \r"+ str(new_data))
        #self.printAiInfo()

        prog = self.makeProg(new_data)
        self.answers.append(prog)
        return prog
    
    def makeProg(self, data):
        layer_0 = np.array(data)
        if(len(data) == 0):
            print()
            print("oops!prog")
            return 1
        layers = self.calcLayers(layer_0)
        return layers[-1]

    def calcLayers(self, layer_0):
        layer_1 = self.activation_func(np.dot(layer_0, self.weights_01))
        layer_2 = self.activation_func(np.dot(layer_1, self.weights_12))
        layer_3 = self.activation_func(np.dot(layer_2, self.weights_23))
        layer_4 = self.output_func(np.dot(layer_3, self.weights_34))
        return (layer_1,layer_2,layer_3,layer_4)
    
    def printAiInfo(self):
        sys.stdout.write("                                                                                                                  \r"\
            +"SBL: " + str(len(self.successful_batches))\
                +" | MAX: " + str(self.max_score)\
                    +" | ERR: " + str(self.avg_error)\
                )
        return

    def iterateCycle(self, is_dead, score):
        self.printAiInfo()

        if(len(self.data) == 0):
            print()
            print("oops!train")
            return
            
        self.trainBatch(not is_dead, score, self.data, self.answers)

        # if is_dead and len(self.successful_batches) >= 30:
        #     for _ in range(self.teach_tries):
        #         for (success_data, success_answer) in self.successful_batches:
        #             self.trainBatch(False, 0.1, success_data[10::2], success_answer[10::2])

        if score > 1:
            self.successful_batches.append( (self.data.copy(), self.answers.copy()) )

        if len(self.successful_batches) >= self.epoch_size:
            print("big training started!")
            for _ in range(self.epochs_count):
                for (success_data, success_answer) in self.successful_batches:
                    self.trainBatch(True, 0.1, success_data[10:], success_answer[10:])

            self.successful_batches.clear()
            print("big training ended!")

            

        self.data.clear()
        self.answers.clear()

    def trainBatch(self, is_positive, score, train_data, train_answers):
        layer_output_error = 0
        
        batch_len = len(train_answers)
        batch_len = batch_len if batch_len > 0 else 1
        magic = np.log((batch_len/13)**2)
        magic = magic if magic > 0 else 1

        if score > self.max_score:
            self.max_score = score

        score = 1 + score / 100
        endorse_power = magic * score# if is_positive else score
        
        #print(batch_len)
        #print(error_power)
        #print(endorse_power)
        for i in range(10 if is_positive else 0, batch_len): #10 first are stubbed data
            
            layer_0 = np.array(train_data[i:i+1])
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

            goal = train_answers[i:i+1][0]
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

            amp =  self.learn_rate* endorse_power          
            if (not (is_positive and ((i % 10) != 0))):
                self.weights_34 -= amp * layer_3.T.dot(layer_4_delta)
                self.weights_23 -= amp * 0.01 * layer_2.T.dot(layer_3_delta)
                self.weights_12 -= amp * 0.01 * layer_1.T.dot(layer_2_delta)
                self.weights_01 -= amp * 0.001 * layer_0.T.dot(layer_1_delta)
                

        # if (is_positive):
        #     self.learn_rate -= self.learn_rate / (batch_len if batch_len > 100 else 100 / 10) / 1000
        #     print(self.learn_rate)

        #print(self.answers)
        #print("error: " + str(layer_3_error))
        self.avg_error = np.average([self.avg_error, layer_output_error])
    
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
        self.gap = 400  #300 ez #130 hard
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
        self.prevGameInfo = deque(maxlen=10)
        self.resetGameInfo()
        self.lastAiCommand = 0.
        self.framerate = 10000
        self.avgScore = 0

    def updateWalls(self):
        self.wallx -= 2
        if self.wallx < -80:
            

            if not self.dead:
                self.counter += 1
                if (self.counter % 20 == 0) and self.gap > 130:
                    self.gap -= 10
                self.avgScore = np.average([self.counter, self.avgScore])
                self.iterateAiCycle(False)

                #self.resetGameInfo()

            self.wallx = 400
            self.offset = random.randint(-110, 110)

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
            self.bird[1] = 350
            self.birdY = 350
            self.dead = False
            # if self.counter > 0:
            #     print("score: " + str(self.counter))
            self.counter = 0
            self.wallx = 400
            self.offset = random.randint(-110, 110)
            self.gravity = 5

            self.iterateAiCycle(True)
            #self.resetGameInfo()
            
    def resetGameInfo(self):
        self.prevGameInfo.clear()
        for i in range(self.prevGameInfo.maxlen):
            self.prevGameInfo.append(self.getGameInfoForAi(i*2))

    def iterateAiCycle(self, is_dead):
        self.iteration += 1
        if self.avgScore < 50 or self.gap > 180:
            self.ai.iterateCycle(is_dead, self.counter)
        every = 300
        if ((self.iteration % every == 0) and (self.avgScore < self.iteration / (every * 2))):
            if (self.iteration > 10000 and self.avgScore < 30) or self.iteration <= 10000:
                self.ai.createNewAi()
                self.avgScore = 0
            else:
                print("nice AI, continue")
            self.iteration = 0

    def updateAi(self):
        self.frame += 1
        if (self.frame >= self.framerate):
            self.frame = 0
        
        if((self.frame % 1) == 0 and not self.dead):
            new_info = self.getGameInfoForAi(0)
            self.prevGameInfo.append(new_info)
            self.lastAiCommand = self.ai.addGameData(np.array(list(self.prevGameInfo)).flatten())
            

    def getGameInfoForAi(self, frame_off):
        return [self.birdY / 600, (self.wallx - (frame_off * 2) + 200) / 800,  (self.offset + self.gap / 2 + 100) / 300, (self.offset - self.gap / 2 + 100) / 300, ]

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

            # if (event.type == pygame.MOUSEBUTTONDOWN) and not self.dead:
            #     self.jump = 17
            #     self.gravity = 5
            #     self.jumpSpeed = 10

            
            if (event.type == pygame.KEYDOWN):
                if event.key == pygame.K_UP:
                    self.framerate = 1000
                elif event.key == pygame.K_DOWN:
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
            self.screen.blit(font.render(str(self.counter) + " | " + str(self.avgScore),
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