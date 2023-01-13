import pygame

class Bullet:
    def __init__(self, x, y, direction=(1, 0)):
        self.x = x
        self.y = y
        self.direction = direction
        self.hit = False

    def update(self):
        # Update the bullet's position based on the direction vector
        self.x += self.direction[0]
        self.y += self.direction[1]

    def draw(self, screen):
        # Draw the bullet to the screen
        pygame.draw.circle(screen, (0, 0, 0), (self.x, self.y), 5)

    def collides_with(self, other):
        # Check if the bullet collides with another object
        return (self.x - other.x) ** 2 + (self.y - other.y) ** 2 < 4
