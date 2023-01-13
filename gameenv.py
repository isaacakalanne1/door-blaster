from player import Player
from enemy import  Enemy
from door import Door

import pygame

class GameEnv:
    def __init__(self):
        self.screen_width = 640
        self.screen_height = 480
        self.player1 = Player(start_pos=(200, 200), color=(255, 0, 0), left_key=pygame.K_a, right_key=pygame.K_d,
                              up_key=pygame.K_w,
                              down_key=pygame.K_s, screen_width=self.screen_width, screen_height=self.screen_height,
                              shoot_key=pygame.K_SPACE, can_collect_ammo=False)
        self.player2 = Player(start_pos=(300, 300), color=(0, 0, 255), left_key=pygame.K_LEFT, right_key=pygame.K_RIGHT,
                              up_key=pygame.K_UP,
                              down_key=pygame.K_DOWN, screen_width=self.screen_width, screen_height=self.screen_height,
                              can_collect_ammo=True)
        self.players = [self.player1, self.player2]
        self.enemies = []
        self.enemy = Enemy(screen_height=self.screen_height, screen_width=self.screen_width, enemies=self.enemies)
        self.ammo_packs = []
        self.door = Door(screen_height=self.screen_height, screen_width=self.screen_width)

    def reset(self):
        self.player1.x, self.player1.y = 200, 200
        self.player2.x, self.player2.y = 300, 300
        self.enemies.clear()
        self.ammo_packs.clear()
        self.door.health = 100
        return [self.player1.x, self.player1.y, self.player2.x, self.player2.y, *[e.x for e in self.enemies],
                *[e.y for e in self.enemies], *[a.x for a in self.ammo_packs], *[a.y for a in self.ammo_packs]]

    def step(self, action):
        self.player1.move(action[:4])
        self.player1.shoot(action[4:])
        self.player2.move(action[4:])
        done = self.door.health == 0
        reward = -1 if done else 1
        return [self.player1.x, self.player1.y, self.player2.x, self.player2.y,
                *[e.x for e in self.enemies],
                *[e.y for e in self.enemies],
                *[a.x for a in self.ammo_packs],
                *[a.y for a in self.ammo_packs]],\
            reward,\
            done,\
            None

    def run_game_loop(self):
        # Initialize Pygame
        pygame.init()
        # Set up the Pygame window
        screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        # Create a clock object to control the framerate
        clock = pygame.time.Clock()
        running = True
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            # Clear the screen
            screen.fill((0, 0, 0))
            # Draw the players, enemies, ammo packs, and door
            self.player1.draw(screen)
            self.player2.draw(screen)
            for enemy in self.enemies:
                enemy.draw(screen)
            for ammo_pack in self.ammo_packs:
                ammo_pack.draw(screen)
            self.door.draw(screen)
            # Update the players, enemies, ammo packs, and door
            self.player1.update(self.ammo_packs, self.player2)
            self.player2.update(self.ammo_packs, self.player1)
            for enemy in self.enemies:
                enemy.update()
            for ammo_pack in self.ammo_packs:
                ammo_pack.update()
            self.door.update(self.player1.bullets + self.player2.bullets)
            # Check for collisions
            for enemy in self.enemies:
                if self.player1.is_colliding(enemy) or self.player2.is_colliding(enemy):
                    running = False
            for ammo_pack in self.ammo_packs:
                if self.player2.is_colliding(ammo_pack):
                    self.player2.ammo += ammo_pack.ammo
                    self.ammo_packs.remove(ammo_pack)
            if self.player1.is_colliding(self.door) and self.door.health == 0:
                running = False
            # Update the screen
            pygame.display.flip()
            # Wait for the next frame
            clock.tick(60)
        # Quit Pygame
        pygame.quit()

