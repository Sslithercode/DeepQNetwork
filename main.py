import numpy as np
import keras
import pygame
import random
from gym import Env
from gym.spaces import Discrete, Box

pygame.init()
screen = pygame.display.set_mode((500,500))

clock = pygame.time.Clock()


class Paddle:
    def __init__(self,pos):
        self.rect = pygame.Rect(pos[0],pos[1],15,100)
        self.vel = pygame.Vector2()
        self.vel.x,self.vel.y = 0,0
    def move(self,keys):
        if keys[pygame.K_w]:
            self.vel.y  = -5
        elif keys[pygame.K_s]:
            self.vel.y =  5
        else:
            self.vel.y  = 0 

        self.rect.y += self.vel.y 

        if self.rect.y < 0:
            self.rect.y = 0
        if self.rect.y + self.rect.height > 500:
            self.rect.y = 500-self.rect.height
    def  draw(self):
        pygame.draw.rect(screen,(255,255,255),self.rect)


class PaddleAgent:
    def __init__(self,pos):
        self.rect = pygame.Rect(pos[0],pos[1],15,100)
        self.vel = pygame.Vector2()
        self.vel.x,self.vel.y = 0,0
    def coord_y(self):
        return self.rect.y
    def move(self,action):
        if action == 0:
            self.vel.y  = -5
        elif action == 1:
            self.vel.y =  5
        else:
            self.vel.y  = 0 
        if self.rect.y < 0:
            self.rect.y = 0
        if self.rect.y + self.rect.height > 500:
            self.rect.y = 500-self.rect.height

        self.rect.y += self.vel.y 
    def  draw(self):
        pygame.draw.rect(screen,(255,255,255),self.rect)

class Ball:
    def __init__(self,pos):
        self.col = (255,255,255)
        self.rect = pygame.Rect(pos[0],pos[1],20,20)
        self.vel = pygame.Vector2()
        self.vel.x, self.vel.y = 5, random.random()*3
    def coord_x(self):
        return self.rect.x
    def coord_y(self):
        return self.rect.y
    def move(self):
        if self.rect.y < 0:
            self.vel.y *= -1
        if self.rect.y + self.rect.height > 500:
            self.vel.y *= -1

        if self.rect.x < 0:
            self.vel.x *= -1
        if self.rect.x + self.rect.width > 500:
            self.vel.x *= -1
        self.rect.y += self.vel.y 
        self.rect.x += self.vel.x


        
        if self.rect.colliderect(paddle_1.rect):
            self.vel.x *= -1
            distance  = (40 + paddle_1.rect.y)  - self.rect.y
            if distance > 0:
                
                self.vel.y = (distance/20+2)*-1
                           
            elif distance < 0:
                self.vel.y = (abs(distance)/20+2)
                
            else:
                self.vel.y = self.vel.y * -1

              
            
        if self.rect.colliderect(paddle_2.rect):
            self.vel.x *= -1
            distance  = (40 + paddle_2.rect.y) - self.rect.y
            if distance > 0:
                
                self.vel.y = (distance/25+2)*-1
                          
            elif distance < 0:
                self.vel.y = (abs(distance)/25+2)
                
            else:
                self.vel.y = self.vel.y * -1
    def draw(self):
        pygame.draw.rect(screen,self.col,self.rect)


class pong_env(Env):
    def __init__(self):
        # create the game
        self.ball = Ball((250,250))
        self.agent = PaddleAgent((460,200))
        self.state = self.ball.coord_x(), self.ball.coord_y(), self.agent.coord_y()
        # Actions, up, down, leave
        self.action_space = Discrete(3)
        # The x and y coordinate of the ball
        self.observation_space = Box(low = np.array([0,0]), high = np.array([500,500]))
        
    def step(self, action):
        self.agent.move(action)
        self.ball.move()
        self.state = [self.ball.coord_x(), self.ball.coord_y(), self.agent.coord_y()]
        if self.ball.rect.colliderect(self.agent.rect):
            reward = 1
        else:
            reward = 0
        if (self.ball.coord_x() < 10 or self.ball.coord_x() > 490):
            done = True
        else:
            done = False

        info = {}
        return self.state, reward, done, info

    def render(self):
        run = True
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
            keys = pygame.key.get_pressed()
            screen.fill((0,0,0))
            paddle_1.draw()
            paddle_1.move(keys)
            self.agent.draw()
            self.ball.draw()
            self.ball.move()
            pygame.display.update()
            clock.tick(60)
        pygame.quit()

    def reset(self):
        del self.ball
        del self.agent
        self.ball = Ball((250,250))
        self.agent = PaddleAgent((460,200))
        self.state = [self.ball.coord_x(), self.ball.coord_y(), self.agent.coord_y()]
        return self.state

paddle_1 = Paddle((25,200))
paddle_2 = Paddle((460,200))
ball = Ball((250,250))


def main():
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        keys  = pygame.key.get_pressed() 
        screen.fill((0,0,0))
        paddle_1.draw()
        paddle_2.draw()
        paddle_1.move(keys)
        paddle_2.move(keys)
        ball.draw()
        ball.move()
        pygame.display.update()
        clock.tick(60)
        print(ball.coord_x())
        
    pygame.quit()

env = pong_env()
states = env.observation_space.shape
actions = env.action_space.n

episodes = 10
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0
    while not done:
        #env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score += reward
    print("Episode:{} Score:{}".format(episode, score))
'''
if __name__ == "__main__":
    main()  
'''


