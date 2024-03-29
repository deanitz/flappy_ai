import pygame
from pygame.locals import *  # noqa
import os, sys
import random
import numpy as np
from collections import deque

np.random.seed(1)

class AI:
    def __init__(self):
        self.data = list()
        self.answers = list()

        self.hidden1_size = 512
        self.hidden2_size = 32
        self.hidden3_size = 16
        #self.hidden4_size = 64
        #self.hidden5_size = 12
        self.input_size = 5
        self.output_size = 1

        self.maxScore = 0
        self.trainOn = True
        self.iterations_between_training = 300
        self.iteration_training = 0
        self.successful_batches = deque(maxlen=100)
        self.failed_batches = deque(maxlen=50)
        self.createNewAi()

        self.activation_func = self.tanh
        self.activation_func_deriv = self.tanh2deriv
        self.output_func = lambda x: x
        self.error_func = self.avg_square_error
        
        self.weights_01_prev = 2 * np.random.random((self.input_size, self.hidden1_size)) - 1
        self.weights_12_prev = 2 * np.random.random((self.hidden1_size,self.hidden2_size)) - 1
        self.weights_23_prev = 2 * np.random.random((self.hidden2_size,self.hidden3_size)) - 1
        self.weights_34_prev = 2 * np.random.random((self.hidden3_size,self.output_size)) - 1
        #self.weights_45_prev = 2 * np.random.random((self.hidden4_size,self.hidden5_size)) - 1
        #self.weights_56_prev = 2 * np.random.random((self.hidden5_size,self.output_size)) - 1

    def createNewAi(self):
        
        self.learn_rate = 0.00005
        self.teach_tries = 1
        
        self.best_batch = deque(maxlen=1)
        self.avg_error = 0

        print()
        if(self.maxScore > 2):    
            print("restoring successful AI")
            self.weights_01 = self.weights_01_prev
            self.weights_12 = self.weights_12_prev
            self.weights_23 = self.weights_23_prev
            self.weights_34 = self.weights_34_prev
            #self.weights_45 = self.weights_45_prev
            #self.weights_56 = self.weights_56_prev
        else:
            print("creating new AI")
            self.weights_01 = 0.2*np.random.random((self.input_size, self.hidden1_size)) - 1
            self.weights_12 = 1*np.random.random((self.hidden1_size,self.hidden2_size)) - 0.5
            self.weights_23 = 1*np.random.random((self.hidden2_size,self.hidden3_size)) - 0.5
            self.weights_34 = 2*np.random.random((self.hidden3_size,self.output_size)) - 1
            #self.weights_45 = np.random.random((self.hidden4_size,self.hidden5_size)) - 0.5
            #self.weights_56 = np.random.random((self.hidden5_size,self.output_size)) - 0.5

        self.printAiInfo()

        self.maxScore = 0

    def save(self):
        print()
        print("saving...")
        path = os.path.join(os.getcwd(), "Weights", "Weights")
        np.save(path, np.array([self.weights_01, self.weights_12, self.weights_23, self.weights_34]), allow_pickle=True)
        print("saving to " + str(path) + " completed")
    
    def load(self):
        print()
        print("loading...")
        path = os.path.join(os.getcwd(), "Weights", "Weights.npy")
        loaded_data = np.load(path, allow_pickle=True)
        self.weights_01 = loaded_data[0]
        self.weights_12 = loaded_data[1]
        self.weights_23 = loaded_data[2]
        self.weights_34 = loaded_data[3]
        print("loading from " + str(path) + " completed")

    def addGameData(self, new_data):
        prog = self.makeProg(new_data)
        if (len(prog) > 0):
            self.data.append(new_data)
            self.answers.append(prog)
            return prog
        else:
            print("oops!prog")
            return 0
    
    def makeProg(self, data):
        layer_0 = np.array(data)
        layers = self.calcLayers(layer_0)
        return layers[-1]

    def calcLayers(self, layer_0):
        if layer_0.shape[0]  == 0:
            return ()
        layer_1 = self.activation_func(np.dot(layer_0, self.weights_01))
        layer_2 = self.activation_func(np.dot(layer_1, self.weights_12))
        layer_3 = self.activation_func(np.dot(layer_2, self.weights_23))
        #layer_4 = self.activation_func(np.dot(layer_3, self.weights_34))
        #layer_5 = self.activation_func(np.dot(layer_4, self.weights_45))
        layer_4 = self.output_func(np.dot(layer_3, self.weights_34))
        return (layer_1,layer_2,layer_3,layer_4)
    
    def printAiInfo(self):
        sys.stdout.write("                                                                                                                  \r"\
            +"SBL: " + str(len(self.successful_batches))\
                +" | FBL: " + str(len(self.failed_batches))\
                    +" | MAX: " + str(self.maxScore)\
                        +" | ERR: " + str(self.avg_error)\
                            +" | ITE: " + str(self.iteration_training)\
                )
        return

    def iterateCycle(self, is_dead, score):

        self.printAiInfo()

        kickstart = score == 0 and self.maxScore < 3
        if kickstart:
            self.trainBatch(not is_dead, score, kickstart, self.data[::5 if kickstart else 1], self.answers[::5 if kickstart else 1])

        if(len(self.answers) > len(self.best_batch)):
                self.best_batch.append( (self.data.copy(), self.answers.copy()) )

        if (score > self.maxScore):
            self.maxScore = score

        if is_dead:
            if 1 <= score:
                self.failed_batches.append( (self.data[::].copy(), self.answers[::].copy(), score) )
                self.failed_batches.append( (self.data[::5].copy(), self.answers[::5].copy(), score) )
            elif score == 0:
                self.failed_batches.append( (self.data[::5].copy(), self.answers[::5].copy(), score) )
            self.iteration_training += 5 if kickstart else 1
            if self.iteration_training > self.iterations_between_training:
                self.iteration_training = 0
                if self.trainOn:
                    for _ in range(self.teach_tries):
                        for (fail_data, fail_answer, fail_score) in self.failed_batches:
                            self.trainBatch(False, 1 / len(self.failed_batches) / 4, fail_score == 0, fail_data, fail_answer)
                        for (success_data, success_answer, successful_score) in self.successful_batches:
                            self.trainBatch(True, successful_score / len(self.successful_batches), False, success_data, success_answer)

            # if(len(self.successful_batches) == 0):
            #     for (success_data, success_answer) in self.best_batch:
            #         self.trainBatch(False, 1, success_data, success_answer)
        else:
            self.iteration_training += 5
            if (self.iteration_training % 5 == 0 and score == 1) or score > 1:
                self.successful_batches.append( (self.data.copy(), self.answers.copy(), score) )

        self.data.clear()
        self.answers.clear()
        #print("dead" if is_dead else "alive")

    def trainBatch(self, is_positive, score, kickstart, train_data, train_anwers):
        layer_output_error = 0
        
        batch_len = len(train_anwers)
        batch_len = batch_len if batch_len > 0 else 1
        magic = (np.log((batch_len/13)**2))
        endorse_power = magic * (1 + score / 100) if is_positive else  1 + score / 100
        #print(endorse_power)

        amp = self.learn_rate if score == 0 else self.learn_rate / 2
        amp *= endorse_power
        
        #print(batch_len)
        #print(error_power)
        #print(endorse_power)
        for i in range(batch_len): #10 first are stubbed data
            
            layer_0 = np.array(train_data[i:i+1])
            layers =  self.calcLayers(layer_0)

            if (len(layers) <= 0):
                print("oops!train")
                return

            layer_1 = layers[0]
            layer_2 = layers[1]
            layer_3 = layers[2]
            layer_4 = layers[3]
            #layer_5 = layers[4]
            #layer_6 = layers[5]

            if not is_positive or is_positive and not i % 2 == 0:
                dropout_mask = np.random.randint(2,size=layer_1.shape)
                layer_1 *= dropout_mask * 2
                
                dropout_mask = np.random.randint(2,size=layer_2.shape)
                layer_2 *= dropout_mask * 2
                
                dropout_mask = np.random.randint(2,size=layer_3.shape)
                layer_3 *= dropout_mask * 2
                
                # dropout_mask = np.random.randint(2,size=layer_4.shape)
                # layer_4 *= dropout_mask * 2
                
                # dropout_mask = np.random.randint(2,size=layer_5.shape)
                # layer_5 *= dropout_mask * 2

            

            goal = train_anwers[i:i+1][0]
            if not is_positive:
                goal = -goal if kickstart else 2*np.random.rand(1)[0] - 1 * goal

            layer_output_error += self.error_func(layer_4, [goal])
            
            #производная от среднеквадратического отклонения - скаляр
            layer_4_delta = (layer_4 - goal)
            
            #back propagation func:
            #поэлементное произведение весов последнего скрытого слоя (рез. - вектор длины скрытого слоя)
            #умноженное на производную от функции активации
            #layer_5_delta = layer_6_delta.dot(self.weights_56.T) * self.activation_func_deriv(layer_5)
            #layer_4_delta = layer_5_delta.dot(self.weights_45.T) * self.activation_func_deriv(layer_4)
            layer_3_delta = layer_4_delta.dot(self.weights_34.T) * self.activation_func_deriv(layer_3)
            layer_2_delta = layer_3_delta.dot(self.weights_23.T) * self.activation_func_deriv(layer_2)
            layer_1_delta = layer_2_delta.dot(self.weights_12.T) * self.activation_func_deriv(layer_1)

            if (score > self.maxScore):
                self.weights_01_prev = self.weights_01
                self.weights_12_prev = self.weights_12
                self.weights_23_prev = self.weights_23
                self.weights_34_prev = self.weights_34
                #self.weights_45_prev = self.weights_45
                #self.weights_56_prev = self.weights_56

                self.maxScore = score

            #self.weights_56 -= amp * layer_5.T.dot(layer_6_delta)
            #self.weights_45 -= (amp / 50) * layer_4.T.dot(layer_5_delta)
            self.weights_34 -= (amp) * layer_3.T.dot(layer_4_delta)
            self.weights_23 -= (amp * 0.5) * layer_2.T.dot(layer_3_delta)
            self.weights_12 -= (amp * 0.5) * layer_1.T.dot(layer_2_delta)
            self.weights_01 -= (amp * 0.05) * layer_0.T.dot(layer_1_delta)

        #print("learn: " + str(self.learn_rate))
        #print("error: " + str(layer_output_error))
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
        self.gap = 200  #300 ez #130 hard real game
        self.wallx = 400
        self.birdY = 350
        self.jump = 0
        self.jumpSpeed = 10
        self.gravity = 5
        self.dead = False
        self.sprite = 0
        self.counter = 0
        self.fitness = 0
        self.offset = random.randint(-110, 110)

        self.ai = AI()
        self.frame = 0
        self.iteration = 0
        self.prevGameInfo = deque(maxlen=1)
        self.resetGameInfo()
        self.lastAiCommand = 0.
        self.framerate = 60
        self.maxScore = 0

    def updateWalls(self):
        self.wallx -= 2
        self.fitness += 1
        if self.wallx < -80:
            self.wallx = 400
            self.offset = random.randint(-110, 110)

            if not self.dead:
                self.counter += 1
                self.ai.printAiInfo()
                if self.counter%30 == 0 and self.gap > 130:
                    self.gap -= 10
                self.maxScore = np.max([self.counter, self.maxScore])
                self.iterateAiCycle(False)
                # !!!!!
                #self.prevGameInfo = deque([self.getGameInfoForAi(0),self.getGameInfoForAi(2)])

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

            self.iterateAiCycle(True)

            self.bird[1] = 50
            self.birdY = 350
            self.dead = False
            self.counter = 0
            self.fitness = 0
            self.wallx = 400
            self.offset = random.randint(-110, 110)
            self.gravity = 5

            

    def iterateAiCycle(self, is_dead):
        self.iteration += 1
        self.ai.iterateCycle(is_dead, self.counter)
        every = 300
        if (self.iteration % every == 0) and (self.maxScore < (self.iteration / every)):
            self.ai.createNewAi()
            self.iteration = 0
            self.maxScore = 0
        

    def updateAi(self):
        self.frame += 1
        if (self.frame >= self.framerate):
            self.frame = 0
        
        if((self.frame % 1) == 0 and not self.dead):
            new_info = self.getGameInfoForAi(0)
            self.prevGameInfo.append(new_info)

            self.lastAiCommand = self.ai.addGameData(np.array(list(self.prevGameInfo.copy())).flatten())

    def resetGameInfo(self):
        self.prevGameInfo.clear()
        for i in range(self.prevGameInfo.maxlen):
            self.prevGameInfo.append(self.getGameInfoForAi(i*2))
            

    def getGameInfoForAi(self, wall_x_fix):
        dist = self.wallx + self.wallUp.get_width()
        if dist - self.bird.left:
            dist = self.bird.left
        return [(self.birdY) / 1000,  (self.birdY - (self.offset + self.gap / 2)) / 1000, (self.offset + self.gap) / 1000,  (self.offset) / 1000, dist / 1000]

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

            
                if (event.type == pygame.KEYUP):
                    if event.key == pygame.K_UP:
                        self.framerate = 3600
                    elif event.key == pygame.K_DOWN:
                        self.framerate = 60
                    elif event.key == pygame.K_s:
                        self.ai.save()
                    elif event.key == pygame.K_l:
                        self.ai.load()
                    elif event.key == pygame.K_TAB:
                        self.ai.trainOn = not self.ai.trainOn
                        if not self.ai.trainOn:
                            self.ai.successful_batches.clear()
                            self.ai.failed_batches.clear()
                            self.ai.maxScore = 0
                        print()
                        print("AI TRAINING" + ("ON" if self.ai.trainOn else "OFF"))

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
            self.screen.blit(font.render(str(self.counter) + " | " + str(self.gap),
                                         -1,
                                         (255, 255, 255)),
                             (160, 50))
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
    print(os.path.join(os.getcwd(), "Weights"))
    FlappyBird().run()