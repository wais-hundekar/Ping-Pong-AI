import pygame
import math
import neat
import os

pygame.init()

FPS = 60

screen = pygame.display.set_mode((1200, 800))
clock = pygame.time.Clock()

leftPaddleMovementChange = 45
rightPaddleMovementChange = 45

colours =  (230, 25, 75) ,  (60, 180, 75) ,  (255, 225, 25) ,  (0, 130, 200) ,  (245, 130, 48) ,  (145, 30, 180) ,  (70, 240, 240) ,  (240, 50, 230) ,  (210, 245, 60) ,  (250, 190, 212) ,  (0, 128, 128) ,  (220, 190, 255) ,  (170, 110, 40) ,  (255, 250, 200) ,  (128, 0, 0) ,  (170, 255, 195) ,  (128, 128, 0) ,  (255, 215, 180) ,  (0, 0, 128) ,  (128, 128, 128) ,  (255, 255, 255) ,  (104, 255, 39) 

class Player:
    height = 100
    width = 15
    y = 350
    def __init__(self, x, colour):
        self.x = x
        self.colour = colour
        if x == 1165:
            self.side = "right"
        elif x == 20:
            self.side = "left"

    def moveUp(self):
        change = 0
        change -= 10
        if self.y < 700 and change > 0 or self.y > 0 and change < 0:       
            self.y += change
    
    def moveDown(self):
        change = 0
        change += 10
        if self.y < 700 and change > 0 or self.y > 0 and change < 0:       
            self.y += change
    
    def moving(self):
        change = 0
        keys = pygame.key.get_pressed() 
        if self.x == 20:
            if keys[pygame.K_w]:
                change -= 10
            if keys[pygame.K_s]:
                change += 10

        elif self.x == 1165:
            if keys[pygame.K_UP]:
                change -= 10
            if keys[pygame.K_DOWN]:
                change += 10

        if self.y < 700 and change > 0 or self.y > 0 and change < 0:       
            self.y += change
    
    def collision(self, ball):
        if ball.x > self.x - 15 and ball.x < self.x + self.width:
            if ball.y > self.y - 15 and ball.y < self.y + self.height + 15:
                return True

    def drawPlayer(self):
        pygame.draw.rect(screen, self.colour, pygame.Rect(self.x, self.y, self.width, self.height))



class Ball:
    
    height, width = 15, 15
    x = 592
    y = 392

    def __init__(self, colour):
        self.colour = colour
        self.xSpeed = 8
        self.ySpeed = self.angleFinder(22)

    def angleFinder(self, angle):
        C = 90 - angle
        a = (math.sin(math.radians(angle))/ math.sin(math.radians(C))) * self.xSpeed
        return a

    def moveBall(self):
        self.x += self.xSpeed
        self.y += self.ySpeed

    def hitWall(self):
        if self.y <= 0 or self.y >= 800 - 15:
            self.ySpeed = self.ySpeed * -1
    
    def bouncing(self, playerY, side):
        ballSpeed = 10
        playeroneYCenter = playerY + 50
        ballYCenter = self.y + 8
        sinOfRight = math.sin(math.radians(90))
        sinOfBallToPlayerY = math.sin(math.radians((playeroneYCenter - ballYCenter) * 1.5))
        sinOfBallToPlayerX = math.sin(math.radians(90 - (playeroneYCenter - ballYCenter) * 1.5))
        self.ySpeed = (sinOfBallToPlayerY/ sinOfRight) * ballSpeed
        if side == "left":
            self.xSpeed = abs((sinOfBallToPlayerX/ sinOfRight) * ballSpeed)
        elif side == "right":
            self.xSpeed = -1 * abs((sinOfBallToPlayerX/ sinOfRight) * ballSpeed)

    def drawBall(self):
        pygame.draw.rect(screen, self.colour, pygame.Rect(self.x, self.y, self.width, self.height))



def main(genomes, config):
    nets = []
    ge = []
    players = []
    balls = []
    genomesList = []
    netsList = []
    playersList = []
    rightSide = False

    testingBallCollisionTimers = []
    testingBallCollisions = []

    colourCounter = 0
    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)

        netsList.append(net)
        if len(netsList) >= 2:
            nets.append(netsList)
            netsList = []


        if rightSide == True:
            playersList.append(Player(1165, colours[colourCounter]))
            rightSide = False
            balls.append(Ball(colours[colourCounter]))
            colourCounter += 1
        elif rightSide == False:
            playersList.append(Player(20, colours[colourCounter]))
            rightSide = True
        if len(playersList) >= 2:
            players.append(playersList)
            playersList = []

        

        g.fitness = 0
        genomesList.append(g)
  
        if len(genomesList) >= 2:
            ge.append(genomesList)
            genomesList = []
            


    running = True
    while running:
        clock.tick(FPS)
        if len(players) <= 0:
            running = False
            break

        for i in range(len(testingBallCollisionTimers)):
            if testingBallCollisions[i] == False and testingBallCollisionTimers[i] == 0:
                testingBallCollisions[i] = True
            elif testingBallCollisionTimers[i] > 0 and testingBallCollisions[i] == False:
                testingBallCollisionTimers[i] -= 1

        for event in pygame.event.get():

                # If the event is quit then exit
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        for i, player in enumerate(players):
            for x, paddle in enumerate(player):

                output = nets[i][x].activate((paddle.y, paddle.x, balls[i].x, paddle.y - balls[i].y))
        
                if output[0] == 1:
                    paddle.moveUp()
                elif output[0] == -1:
                    paddle.moveDown()


        for i, player in enumerate(players):
            for x, paddle in enumerate(player):
                if paddle.collision(balls[i]):
                    ge[i][x].fitness += 5
                    balls[i].bouncing(paddle.x, paddle.side)
        
        

        screen.fill((0, 0, 0))

        screenYPlace = 5
        while screenYPlace < screen.get_height():
            pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(595, screenYPlace, 10, 10))
            screenYPlace += 20

        rem = []
        playerrem = []
        netsrem = []
        gerem = []
        for x, ball in enumerate(balls):
            if ball.x < 0 or ball.x > 1200:
                playerrem.append(players[x])
                netsrem.append(nets[x])
                gerem.append(ge[x])
                rem.append(ball)

        for ball in rem:
            balls.remove(ball)        
        for player in playerrem:
            players.remove(player)
        for net in netsrem:
            nets.remove(net)
        for soloGe in gerem:
            ge.remove(soloGe)

        for player in players:
            for paddle in player:
                paddle.drawPlayer()
        
        

        for ball in balls:
            ball.hitWall()
            ball.moveBall()
            ball.drawBall()

        pygame.display.update()




def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to 50 generations.
    winner = p.run(main, 50)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))



if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)
