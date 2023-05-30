import numpy as np
import pygame
import random
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow import keras
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import keras.backend as K
pygame.init()
screen = pygame.display.set_mode((500,500))

clock = pygame.time.Clock()

GAMMA  = 0.9

tfd = tfp.distributions



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
        self.network = keras.Sequential([
            keras.layers.Dense(3,input_shape=[3],activation="relu"),
            keras.layers.Dense(2,activation="relu"),
            keras.layers.Dense(3,activation="softmax")
        ])
        self.network.compile(optimizer="adam")
        self.rect = pygame.Rect(pos[0],pos[1],15,100)
        self.vel = pygame.Vector2()
        self.vel.x,self.vel.y = 0,0


        self.rewards = []
        self.actions = []
        self.states = []


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
    
    def act(self,state):
        pred = self.network(np.array([state]))[0]
        probs = [round(float(i),3) for i in pred]
        probs[2] = round(1  - (probs[0]+probs[1]),3)
        action = np.random.choice([0,1,2],1,p=probs)
        return action
    
    def reset_memory(self):
        self.rewards = []
        self.actions = []
        self.states = []


    @tf.function
    def train_step(self, dr, state, action):
        with tf.GradientTape() as tape:
            probs = self.network(state)  # recalculate probs
            dist = tfd.Categorical(probs=probs)
            log_p = dist.log_prob(action)  # convert to log probabilities as in the REINFORCE Paper
            loss =  -tf.cast(dr, tf.float32) * tf.squeeze(log_p)  # Compute loss: discounted_rewards * log_ps but use a negative for gradient ascent
        grad = tape.gradient(loss, self.network.trainable_variables)
        self.network.optimizer.apply_gradients(zip(grad, self.network.trainable_variables))


    def update_params(self):
        actions   = tf.convert_to_tensor(self.actions,dtype=tf.float32)
        rewards = tf.convert_to_tensor(self.rewards,dtype=tf.float32)
        discounted = np.zeros_like(self.rewards)
        _rad = 0
        for t in range(len(rewards)-1,-1,-1):
            if rewards[t] != 0:
                _rad = 0
            _rad *= GAMMA
            _rad +=rewards[t]
            discounted[t] = _rad


     
        for i,(dr,state_m) in enumerate(zip(discounted,self.states)):
            state = tf.convert_to_tensor([state_m]) 
            self.train_step(dr, state, actions[i])
            
            


class Ball:
    def __init__(self,pos):
        self.col = (255,255,255)
        self.rect = pygame.Rect(pos[0],pos[1],20,20)
        self.vel = pygame.Vector2()
        self.vel.x, self.vel.y = random.choice([-5,5]),0
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
    def __init__(self, agent1,agent2):
        # create the game
        self.ball = Ball((250,250))
        self.agent1,self.agent2 =   agent1,agent2
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

       

        
        if (self.ball.coord_x() < 0 or self.ball.coord_x() > 450):
            if self.ball.coord_x() < 0:
                reward2 = -1
                reward1  = 1
            if self.ball.coord_x() > 450:
                reward1 = -1
                reward2 = 1
            done = True
        else:
            done = False
            reward1,reward2 = 0,0

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

    def reset(self):
        self.ball.rect.x, self.ball.rect.y = 250,250
        self.ball.vel.x, self.ball.vel.y =  random.choice([-5,5]), 0
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


player_A , player_B = PaddleAgent((15,200)), PaddleAgent((460,200))
env = pong_env(player_A,player_B)
states = env.observation_space.shape
actions = env.action_space.n


def compute_discount(rewards):
    new_rewards = rewards.copy()
    count  = 0
    for i in range(len(rewards) - 1, -1, -1):
        new_rewards[i] =round(rewards[-1] * (GAMMA ** count),4)
        count += 1
    return new_rewards

def  train():
    episodes = 1000
    for episode in range(1, episodes+1):
        player_A.reset_memory()
        player_B.reset_memory()
        state1,state2 = env.reset()
        done = False
        scores  =  [0,0]
        while not done:
            env.render()
            action1 = player_A.act(state1)
            action2 = player_B.act(state2)
            player_A.states.append(state1)
            player_B.states.append(state2)
            player_A.actions.append(action1)
            player_B.actions.append(action2)
            state1,state2, r1,r2, done, info = env.step(action1,action2)
            player_A.rewards.append(r1)
            player_B.rewards.append(r2)
            scores[0] += r1 
            scores[1] += r2
        print("Episode:{} Agent1_Score:{} Agent2_Score:{}".format(episode, scores[0], scores[1]))
        player_A.update_params()
        player_B.update_params()
train()



