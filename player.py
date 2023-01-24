import pygame
import bullet
import math

Bullet = bullet.Bullet

class Player:
    def __init__(self, start_pos, color, screen_width, screen_height, can_collect_ammo, can_shoot):
        self.x, self.y = start_pos
        self.start_pos = start_pos
        self.color = color
        self.bullets = []
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.can_shoot = can_shoot
        self.can_collect_ammo = can_collect_ammo
        self.ammo = 10
        self.size = 10
        self.last_bullet_time = 0
        self.elapsed_time = 0

    def update(self, action, ammo_packs, other_player):
        reward = 0
        # Update player position based on user input, but don't allow the player to go off the screen
        # print("Action is " + str(action))
        if action == 1 and self.x > 0:
            self.x -= 5
        if action == 2 and self.x < self.screen_width:
            self.x += 5
        if action == 3 and self.y > 0:
            self.y -= 5
        if action == 4 and self.y < self.screen_height:
            self.y += 5

        # if self.is_on_edge_of_screen():
        #     reward = -50

        self.start_pos = (self.x, self.y)

        player_radius = 10

        # Check for intersections with ammo packs
        if self.can_collect_ammo:
            for ammo_pack in ammo_packs:
                if self.x > ammo_pack.x - ammo_pack.radius - player_radius\
                        and self.x < ammo_pack.x + ammo_pack.radius + player_radius\
                        and self.y > ammo_pack.y - ammo_pack.radius - player_radius\
                        and self.y < ammo_pack.y + ammo_pack.radius + player_radius:
                    # Remove the ammo pack and add 10 ammo to the player's ammo count
                    reward += 10
                    ammo_packs.remove(ammo_pack)
                    self.ammo += 10

        if self.can_collect_ammo:
            radius = 15
            if self.x > other_player.x - radius and self.x < other_player.x + radius and self.y > other_player.y - radius and self.y < other_player.y + radius:
                # Transfer all the player's ammo to the other player
                reward += 10
                other_player.ammo += self.ammo
                self.ammo = 0

        self.elapsed_time = pygame.time.get_ticks()

        # Shoot a bullet in the specified direction if the player has ammo and the shoot key is pressed
        if self.can_shoot and self.ammo > 0 and self.elapsed_time - self.last_bullet_time >= 250:

            speed = 3

            if action == 5:
                self.bullets.append(Bullet(x=self.x, y=self.y, direction=(-speed, 0)))
                self.ammo -= 1
                self.last_bullet_time = pygame.time.get_ticks()
            elif action == 6:
                self.bullets.append(Bullet(x=self.x, y=self.y, direction=(0, speed)))
                self.ammo -= 1
                self.last_bullet_time = pygame.time.get_ticks()
            elif action == 7:
                self.bullets.append(Bullet(x=self.x, y=self.y, direction=(speed, 0)))
                self.ammo -= 1
                self.last_bullet_time = pygame.time.get_ticks()
            elif action == 8:
                self.bullets.append(Bullet(x=self.x, y=self.y, direction=(0, -speed)))
                self.ammo -= 1
                self.last_bullet_time = pygame.time.get_ticks()

        # Update bullet positions
        for b in self.bullets:
            b.update()

        return reward

    def is_on_edge_of_screen(self):
        return self.x <= 0 or self.x >= self.screen_width or self.y <= 0 or self.y >= self.screen_height

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
