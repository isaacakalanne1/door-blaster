from player import Player
from enemy import Enemy
from door import Door
from ammopack import AmmoPack
import random
import time
import pygame


class GameEnv:
    def __init__(self):

        pygame.init()
        self.screen_width = 640
        self.screen_height = 480
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

    def get_starting_shooter_obs(self):
        return 0

    def get_starting_collector_obs(self):
        return 0

    def spawn_ammo_pack(self, packs):

        # Generate random coordinates within the screen bounds
        x = random.randint(0, self.screen_width)
        y = random.randint(0, self.screen_height)

        # Don't spawn a new ammo pack if there are already two on the screen
        if len(packs) >= 2:
            return

        # Get the current time
        current_time = time.perf_counter()

        # Spawn a new ammo pack every 5 seconds
        if current_time - self.ammo_pack_timer > 5:
            self.ammo_packs.append(AmmoPack(x, y))
            self.ammo_pack_timer = current_time

    def get_shooter_observation(self):
        return [0]

    def take_game_step(self, s_action, c_action):
        reward = 0
        done = False

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True

        self.player1.update(s_action, self.ammo_packs, self.player2)
        self.player2.update(c_action, self.ammo_packs, self.player1)

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
        if current_time - self.last_spawn_time > random.randint(1000, 1500) and len(self.enemies) < 10:
            self.enemies.append(Enemy(screen_height=self.screen_height,
                                      screen_width=self.screen_width,
                                      enemies=self.enemies))
            self.last_spawn_time = current_time

        for enemy in self.enemies:
            enemy.draw(self.screen)

        self.spawn_ammo_pack(self.ammo_packs)

        for ammo_pack in self.ammo_packs:
            ammo_pack.draw(self.screen)

        self.door.update(self.player1.bullets + self.player2.bullets)

        self.door.draw(self.screen)

        for enemy in self.enemies:
            if self.player1.is_colliding(enemy) or self.player2.is_colliding(enemy):
                # reset the game
                self.reset_game()
                break

        # Check if both players are touching the door
        if self.door.color == (0, 255, 0) \
                and self.door.x - self.door.size < self.player1.x < self.door.x + self.door.size \
                and self.door.y - self.door.size < self.player1.y < self.door.y + self.door.size \
                and self.door.x - self.door.size < self.player2.x < self.door.x + self.door.size \
                and self.door.y - self.door.size < self.player2.y < self.door.y + self.door.size:
            # Reset game objects
            self.reset_game()

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

        return (self.get_shooter_observation(), done), (self.get_shooter_observation(), done)

    def run_game_loop(self):
        # Main game loop
        running = True
        while running:
            self.take_game_step()