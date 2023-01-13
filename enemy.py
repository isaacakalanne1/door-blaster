import pygame
import random

class Enemy:
    def __init__(self, screen_height, screen_width, enemies):
        self.screen_height = screen_height
        self.screen_width = screen_width
        self.enemies = enemies
        self.x, self.y, self.dir = self.get_random_pos()
        self.speed = 3
        self.color = (128, 0, 128)  # purple
        self.size = 20
        self.last_spawn_time = pygame.time.get_ticks()
        self.spawn_interval = random.randint(3000, 5000)  # 3-5 seconds

    def update(self):
        self.x, self.y = self.get_new_pos()
        if self.x < 0 or self.x > self.screen_width or self.y < 0 or self.y > self.screen_height:
            self.enemies.remove(self)

    def get_random_pos(self):
        side = random.randint(1, 4)
        if side == 1:  # Left
            self.x = 0
            self.y = random.randint(0, self.screen_height)
            self.dir = 4  # travel right
        elif side == 2:  # Top
            self.x = random.randint(0, self.screen_width)
            self.y = 0
            self.dir = 2  # travel down
        elif side == 3:  # Right
            self.x = self.screen_width
            self.y = random.randint(0, self.screen_height)
            self.dir = 3  # travel left
        elif side == 4:  # Bottom
            self.x = random.randint(0, self.screen_width)
            self.y = self.screen_height
            self.dir = 1  # travel up
        return self.x, self.y, self.dir

    def get_new_pos(self):
        if self.dir == 1:  # Up
            self.y -= self.speed
        elif self.dir == 2:  # Down
            self.y += self.speed
        elif self.dir == 3:  # Left
            self.x -= self.speed
        elif self.dir == 4:  # Right
            self.x += self.speed
        return self.x, self.y

    def draw(self, screen):
        pygame.draw.polygon(screen, self.color, ((self.x, self.y - self.size),
                                                 (self.x - self.size, self.y + self.size),
                                                 (self.x + self.size, self.y + self.size)))




