import numpy as np
import keras
import pygame
import random
import tensorflow as tf
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
        self.vel.x, self.vel.y = 5, 0
    def coord_x(self):
        return self.rect.x
    def coord_y(self):
        return self.rect.y
    def move(self, paddle1, paddle2):
        c_p1, c_p2  = False,False
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


        
        if self.rect.colliderect(paddle1.rect):
            self.vel.x *= -1
            distance  = (40 + paddle1.rect.y)  - self.rect.y
            if distance > 0:
                
                self.vel.y = (distance/20+2)*-1
                           
            elif distance < 0:
                self.vel.y = (abs(distance)/20+2)
                
            else:
                self.vel.y = self.vel.y * -1

              
        
        if self.rect.colliderect(paddle2.rect):
            c_p2 = True
            self.vel.x *= -1
            distance  = (40 + paddle2.rect.y) - self.rect.y
            if distance > 0:
                
                self.vel.y = (distance/25+2)*-1
                          
            elif distance < 0:
                self.vel.y = (abs(distance)/25+2)
                
            else:
                self.vel.y = self.vel.y * -1
        return c_p2
    def draw(self):
        pygame.draw.rect(screen,self.col,self.rect)


class pong_env(Env):
    def __init__(self):
        # create the game
        self.ball = Ball((250,250))
        self.agent2 = PaddleAgent((460,200))
        self.agent1 = PaddleAgent((15,200))
        self.state1 = [self.ball.coord_x(), self.ball.coord_y(), self.agent1.coord_y()]
        self.state2 = [self.ball.coord_x(), self.ball.coord_y(), self.agent2.coord_y()]
        # Actions, up, down, leave
        self.action_space = Discrete(3)
        # The x and y coordinate of the ball
        self.observation_space = Box(low = np.array([0,0]), high = np.array([500,500]))
        
    def step(self, action1,action2):
        self.agent1.move(action1)
        self.agent2.move(action2)
        self.ball.move(self.agent1, self.agent2)
        self.state1 = [self.ball.coord_x(), self.ball.coord_y(), self.agent1.coord_y()]
        self.state2 = [self.ball.coord_x(), self.ball.coord_y(), self.agent2.coord_y()]

       

        if self.ball.rect.colliderect(self.agent1.rect):
            reward1 = 1
        else:
            reward1 = 0

        if self.ball.rect.colliderect(self.agent2.rect):
            reward2 = 1
        else:
            reward2 = 0

        if (self.ball.coord_x() < 10 or self.ball.coord_x() > 490):
            done = True
        else:
            done = False

        info = {}
        return self.state1,self.state2, reward1, reward2, done, info
    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        keys = pygame.key.get_pressed()
        screen.fill((0,0,0))
        self.agent1.draw()
        self.agent2.draw()
        self.ball.draw()
        pygame.display.update()
        clock.tick(60)

    def reset(self):
        self.ball.rect.x, self.ball.rect.y = 250,250
        self.ball.vel.x, self.ball.vel.y = 5,0
        self.agent1.rect.x, self.agent1.rect.y = 15,200
        self.agent2.rect.x, self.agent2.rect.y = 460,200
        self.state1 = [self.ball.coord_x(), self.ball.coord_y(), self.agent1.coord_y()]
        self.state2 = [self.ball.coord_x(), self.ball.coord_y(), self.agent2.coord_y()]
        return self.state1, self.state2



'''
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
'''
class AgentInfo:
    def __init__(self):
        self.n_state,self.reward,self.score = None,0,0
        
agent1_info = AgentInfo()
agent2_info = AgentInfo()

env = pong_env()
states = env.observation_space.shape
actions = env.action_space.n

episodes = 10
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    agent1_info.score = 0
    agent2_info.score = 0
    while not done:
        env.render()
        action1 = env.action_space.sample()
        action2 = env.action_space.sample()
        agent1_info.n_state, agent2_info.n_state,agent1_info.reward,agent2_info.reward, done, info = env.step(action1,action2)
        agent1_info.score += agent1_info.reward
        agent2_info.score += agent2_info.reward
    print("Episode:{} Agent1_Score:{} Agent2_Score:{}".format(episode, agent1_info.score, agent2_info.score))
'''
if __name__ == "__main__":
    main()  
'''


