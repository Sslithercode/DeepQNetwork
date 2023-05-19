
import keras
import pygame
import random

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
    def move(self,keys):
        if keys[pygame.K_w]:
            self.vel.y  = -5
        elif keys[pygame.K_s]:
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

paddle_1 = Paddle((25,200))
paddle_2 = PaddleAgent((460,200))
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

    pygame.quit()

if __name__ == "__main__":
    main()
