import pygame
import random

class Door:
    def __init__(self, screen_height, screen_width):
        # Choose a random position on the edge of the screen
        side = random.randint(1, 4)
        if side == 1:  # Left
            self.x = 0
            self.y = random.randint(0, screen_height)
        elif side == 2:  # Top
            self.x = random.randint(0, screen_width)
            self.y = 0
        elif side == 3:  # Right
            self.x = screen_width
            self.y = random.randint(0, screen_height)
        elif side == 4:  # Bottom
            self.x = random.randint(0, screen_width)
            self.y = screen_height
        self.size = 20
        self.color = (255, 0, 0)  # Red
        self.max_health = 100
        self.health = self.max_health

    def update(self, bullets):
        reward = 0
        # Check for intersections with bullets
        for bullet in bullets:
            if bullet.hit:
                # Skip bullets that have already hit the door
                continue
            if self.x > bullet.x - self.size and self.x < bullet.x + self.size and self.y > bullet.y - self.size and self.y < bullet.y + self.size:
                # Decrement the door's health and change the color based on the remaining health
                self.health -= 10
                reward += 10
                if self.health > 50:
                    self.color = (255, 0, 0)  # Red
                elif self.health > 20:
                    self.color = (255, 128, 0)  # Orange
                else:
                    self.color = (255, 255, 0)  # Yellow
                # Remove the bullet
                bullet.hit = True

        # Check if the door's health has reached zero
        if self.health <= 0:
            self.color = (0, 255, 0)  # Green
        return reward

    def draw(self, surface):
        # Draw the door
        pygame.draw.circle(surface, self.color, (self.x, self.y), self.size)
