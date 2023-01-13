import pygame

class AmmoPack:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.color = (0, 255, 0)  # Green color
        self.radius = 5

    def draw(self, screen):
        # Draw the ammo pack to the screen
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius)

