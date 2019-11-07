import pygame
from pygame.locals import *  # noqa
import sys
import random
import numpy as np

class AI:
    def __init__(self):
        self.data = list()
        self.answers = list()

        self.learn_rate = 0.0005
        self.hidden_size = 12
        self.input_size = 10

        self.weights_01 = 2 * np.random.random((self.input_size, self.hidden_size)) - 1
        self.weights_12 = 2 * np.random.random((self.hidden_size,1)) - 1

        self.activation_func = self.tanh
        self.activation_func_deriv = self.tanh2deriv
        self.output_func = lambda x: x
        self.error_func = self.avg_square_error

        
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
        return layers[1]

    def calcLayers(self, layer_0):
        layer_1 = self.activation_func(np.dot(layer_0, self.weights_01))
        layer_2 = self.output_func(np.dot(layer_1, self.weights_12))
        return (layer_1,layer_2)
    
    def printAiInfo(self):
        print(len(self.data))

    def iterateCycle(self, is_dead):
        self.trainBatch(not is_dead)
        self.data.clear()
        self.answers.clear()
        print("dead" if is_dead else "alive")

    def trainBatch(self, is_positive):
        layer_2_error = 0
        
        for i in range(len(self.answers)):
            layer_0 = np.array(self.data[i:i+1])
            layers =  self.calcLayers(layer_0)
            layer_1 = layers[0]
            layer_2 = layers[1]
            
            goal = self.answers[i:i+1][0]
            goal = goal if is_positive else (-goal)

            layer_2_error += self.error_func(layer_2, [goal])
            
            #производная от среднеквадратического отклонения - скаляр
            layer_2_delta = layer_2 - goal 
            
            #back propagation func:
            #поэлементное произведение весов последнего скрытого слоя (рез. - вектор длины скрытого слоя)
            #умноженное на производную от функции активации
            layer_1_delta = layer_2_delta * self.weights_12.T * self.activation_func_deriv(layer_1)
            
            error_power = 2 if i == 0 else (1 / np.sqrt(i))

            self.weights_12 -= self.learn_rate * layer_1.T.dot(layer_2_delta) * error_power
            self.weights_01 -= self.learn_rate * layer_0.T.dot(layer_1_delta) * error_power

        print("error: " + str(layer_2_error))
    
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
        self.gap = 130
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
        self.prevGameInfo = self.getGameInfoForAi()
        self.lastAiCommand = 0.

    def updateWalls(self):
        self.wallx -= 2
        if self.wallx < -80:
            self.wallx = 400
            self.offset = random.randint(-110, 110)

            if not self.dead:
                self.counter += 1
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
            self.birdY = 50
            self.dead = False
            self.counter = 0
            self.wallx = 400
            self.offset = random.randint(-110, 110)
            self.gravity = 5

            self.iterateAiCycle(True)

    def iterateAiCycle(self, is_dead):
        self.iteration += 1
        self.ai.iterateCycle(is_dead)

    def updateAi(self):
        self.frame += 1
        if (self.frame >= 60):
            self.frame = 0
        
        if(self.frame%5==0 and not self.dead):
            new_info = self.getGameInfoForAi()
            gameInfo = self.prevGameInfo + new_info
            self.lastAiCommand = self.ai.addGameData(gameInfo)
            self.prevGameInfo = new_info
            

    def getGameInfoForAi(self):
        return [self.birdY / 1000, (self.wallx + 200) / 1000,  (self.offset + 100) / 200, self.gravity / 10, self.jump / 20]

    def run(self):
        clock = pygame.time.Clock()
        pygame.font.init()
        font = pygame.font.SysFont("Arial", 50)
        while True:
            clock.tick(60)

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
            if(self.lastAiCommand > 0.5 and not self.dead):
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