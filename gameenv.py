from player import Player
from enemy import Enemy
from door import Door
from ammopack import AmmoPack
import random
import time
import pygame
import numpy as np


class GameEnv:
    def __init__(self):

        pygame.init()
        self.screen_width = 640
        self.screen_height = 480
        self.max_enemies = 10
        self.max_ammo_packs = 2
        # Set up the Pygame window
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))

        # Create a clock object to control the framerate
        self.clock = pygame.time.Clock()

        # Set up a timer to keep track of elapsed time
        self.ammo_pack_timer = time.perf_counter()

        self.last_spawn_time = pygame.time.get_ticks()

        self.player1 = Player(start_pos=(200, 200), color=(255, 0, 0), screen_width=self.screen_width,
                              screen_height=self.screen_height, can_shoot=True, can_collect_ammo=False)
        self.player2 = Player(start_pos=(300, 300), color=(0, 0, 255), screen_width=self.screen_width,
                              screen_height=self.screen_height, can_shoot=False, can_collect_ammo=True)
        self.players = [self.player1, self.player2]
        self.enemies = []
        self.enemy = Enemy(screen_height=self.screen_height, screen_width=self.screen_width, enemies=self.enemies)
        self.ammo_packs = []
        self.door = Door(screen_height=self.screen_height, screen_width=self.screen_width)
        self.player_1_reward = 0
        self.player_2_reward = 0

    def reset_game(self):
        # Reset game objects
        self.player1 = Player(start_pos=(200, 200), color=(255, 0, 0), screen_width=self.screen_width,
                              screen_height=self.screen_height, can_shoot=True, can_collect_ammo=False)
        self.player2 = Player(start_pos=(300, 300), color=(0, 0, 255), screen_width=self.screen_width,
                              screen_height=self.screen_height, can_shoot=False, can_collect_ammo=True)
        self.players = [self.player1, self.player2]
        self.enemies = []
        self.enemy = Enemy(screen_height=self.screen_height, screen_width=self.screen_width, enemies=self.enemies)
        self.ammo_packs = []
        self.door = Door(screen_height=self.screen_height, screen_width=self.screen_width)
        self.player_1_reward = 0
        self.player_2_reward = 0

    def get_shooter_state(self):
        return self.get_state(ego_player=self.player1, other_player=self.player2)

    def get_collector_state(self):
        return self.get_state(ego_player=self.player1, other_player=self.player2)

    def get_state(self, ego_player, other_player):
        state = [
            # Distance from 4 walls
            ego_player.x,
            self.screen_width - ego_player.x,
            ego_player.y,
            self.screen_height - ego_player.y,
            # Relative position of door
            self.door.x - ego_player.x,
            self.door.y - ego_player.y,
            # Current ammo
            ego_player.ammo,
            # Ammo of other player
            other_player.ammo,
            # Relative position of other player
            other_player.x - ego_player.x,
            other_player.y - ego_player.y,
            # Door health
            self.door.health
        ]

        # Relative position of enemies
        for i in range(self.max_enemies):
            if i < len(self.enemies):
                enemy = self.enemies[i]
                state.append(enemy.x - ego_player.x)
                state.append(enemy.y - ego_player.y)
            else:
                state.append(0)
                state.append(0)

        # Relative distance of ammo packs
        for i in range(self.max_ammo_packs):
            if i < len(self.ammo_packs):
                ammo_pack = self.ammo_packs[i]
                state.append(ammo_pack.x - ego_player.x)
                state.append(ammo_pack.y - ego_player.y)
            else:
                state.append(0)
                state.append(0)

        return np.array(state)

    def spawn_ammo_pack(self, packs):

        # Generate random coordinates within the screen bounds
        x = random.randint(0, self.screen_width)
        y = random.randint(0, self.screen_height)

        # Don't spawn a new ammo pack if there are already two on the screen
        if len(packs) >= self.max_ammo_packs:
            return

        # Get the current time
        current_time = time.perf_counter()

        # Spawn a new ammo pack every 5 seconds
        if current_time - self.ammo_pack_timer > 5:
            self.ammo_packs.append(AmmoPack(x, y))
            self.ammo_pack_timer = current_time

    def take_game_step(self, s_action, c_action):
        self.player_1_reward = 0
        self.player_2_reward = 0
        done = False

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True

        self.player_1_reward += self.player1.update(s_action, self.ammo_packs, self.player2)
        self.player_2_reward += self.player2.update(c_action, self.ammo_packs, self.player1)

        for enemy in self.enemies:
            enemy.update()

        # Check for collisions between player.py bullets and enemies
        for player in self.players:
            for bullet in player.bullets:
                for enemy in self.enemies:
                    if bullet.collides_with(enemy):
                        player.bullets.remove(bullet)

        for player in self.players:
            player.draw(self.screen)
        # Check if it's time to spawn a new enemy
        current_time = pygame.time.get_ticks()
        if current_time - self.last_spawn_time > random.randint(1000, 1500) and len(self.enemies) < self.max_enemies:
            self.enemies.append(Enemy(screen_height=self.screen_height,
                                      screen_width=self.screen_width,
                                      enemies=self.enemies))
            self.last_spawn_time = current_time

        for enemy in self.enemies:
            enemy.draw(self.screen)

        self.spawn_ammo_pack(self.ammo_packs)

        for ammo_pack in self.ammo_packs:
            ammo_pack.draw(self.screen)

        self.player_1_reward += self.door.update(self.player1.bullets + self.player2.bullets)

        self.door.draw(self.screen)

        for enemy in self.enemies:
            if self.player1.is_colliding(enemy):
                # reset the game
                self.player_1_reward = -50
                done = True
                break
            if self.player2.is_colliding(enemy):
                # reset the game
                self.player_2_reward = -50
                done = True
                break

        # Check if both players are touching the door
        if self.is_player_touching_door(self.player1):
            self.player_1_reward = +50
        if self.is_player_touching_door(self.player2):
            self.player_2_reward = +50
        if self.is_player_touching_door(self.player1) and self.is_player_touching_door(self.player2):
            done = True

        # Initialize the font
        font = pygame.font.Font(None, 36)

        # Create a text surface and rect to hold the text
        text_surface = font.render(f"Ammo: {self.player1.ammo}", True, (0, 0, 0))
        text_rect = text_surface.get_rect()

        # Set the text rect to the top left corner of the screen
        text_rect.topleft = (10, 10)

        # Draw the text to the screen
        self.screen.blit(text_surface, text_rect)

        pygame.display.flip()

        # Limit the frame-rate to 60 fps
        # self.clock.tick(60) # Activate this when playing the game at normal speed

        # Draw the game to the screen
        self.screen.fill((255, 255, 255))  # Fill the screen with white

        return (self.get_shooter_state(), self.player_1_reward, done),\
            (self.get_collector_state(), self.player_2_reward, done)

    def is_player_touching_door(self, player):
        if self.door.color == (0, 255, 0) \
                and self.door.x - self.door.size < player.x < self.door.x + self.door.size \
                and self.door.y - self.door.size < player.y < self.door.y + self.door.size:
            return True
