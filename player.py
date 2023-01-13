import pygame
import bullet
import math

Bullet = bullet.Bullet

class Player:
    def __init__(self, start_pos, color, left_key, right_key, up_key, down_key, screen_width, screen_height, can_collect_ammo, shoot_key=None):
        self.x, self.y = start_pos
        self.start_pos = start_pos
        self.color = color
        self.bullets = []
        self.left_key = left_key
        self.right_key = right_key
        self.up_key = up_key
        self.down_key = down_key
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.shoot_key = shoot_key
        self.can_collect_ammo = can_collect_ammo
        self.ammo = 10
        self.size = 10
        self.last_bullet_time = 0
        self.elapsed_time = 0

    def move(self, action):
        # Unpack the action
        left, right, up, down = action
        # Check if the left key is pressed
        if left:
            self.x -= 5
        # Check if the right key is pressed
        if right:
            self.x += 5
        # Check if the up key is pressed
        if up:
            self.y -= 5
        # Check if the down key is pressed
        if down:
            self.y += 5
        # Check if the player goes out of bounds
        self.x = max(0, min(self.x, self.screen_width))
        self.y = max(0, min(self.y, self.screen_height))

        self.start_pos = (self.x, self.y)

    def shoot(self, action):
        # Unpack the action
        left, right, up, down = action

        self.elapsed_time = pygame.time.get_ticks()

        # Check if the player has ammo
        if self.ammo > 0 and self.elapsed_time - self.last_bullet_time >= 250:
            # Check if the left key is pressed
            if left:
                # Create a bullet that moves to the left
                new_bullet = Bullet(self.x, self.y, (-5, 0))
                self.bullets.append(new_bullet)
                self.ammo -= 1
                self.last_bullet_time = pygame.time.get_ticks()
            # Check if the right key is pressed
            if right:
                # Create a bullet that moves to the right
                new_bullet = Bullet(self.x, self.y, (5, 0))
                self.bullets.append(new_bullet)
                self.ammo -= 1
                self.last_bullet_time = pygame.time.get_ticks()
            # Check if the up key is pressed
            if up:
                # Create a bullet that moves up
                new_bullet = Bullet(self.x, self.y, (0, -5))
                self.bullets.append(new_bullet)
                self.ammo -= 1
                self.last_bullet_time = pygame.time.get_ticks()
            # Check if the down key is pressed
            if down:
                # Create a bullet that moves down
                new_bullet = Bullet(self.x, self.y, (0, 5))
                self.bullets.append(new_bullet)
                self.ammo -= 1
                self.last_bullet_time = pygame.time.get_ticks()

    def update(self, ammo_packs, other_player):

        player_radius = 10

        # Check for intersections with ammo packs
        if self.can_collect_ammo:
            for ammo_pack in ammo_packs:
                if self.x > ammo_pack.x - ammo_pack.radius - player_radius\
                        and self.x < ammo_pack.x + ammo_pack.radius + player_radius\
                        and self.y > ammo_pack.y - ammo_pack.radius - player_radius\
                        and self.y < ammo_pack.y + ammo_pack.radius + player_radius:
                    # Remove the ammo pack and add 10 ammo to the player's ammo count
                    ammo_packs.remove(ammo_pack)
                    self.ammo += 10

        if self.can_collect_ammo:
            radius = 15
            if self.x > other_player.x - radius and self.x < other_player.x + radius and self.y > other_player.y - radius and self.y < other_player.y + radius:
                # Transfer all of the player's ammo to the other player
                other_player.ammo += self.ammo
                self.ammo = 0

        # Update bullet positions
        for b in self.bullets:
            b.update()

    def draw(self, screen):
        # Draw the player to the screen
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.size)

        # Draw the player's bullets
        for b in self.bullets:
            b.draw(screen)

    def is_colliding(self, other):
        distance = math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
        if distance < self.size + other.size:
            return True
        else:
            return False
