import pygame
import player
import enemy
import random
import ammopack
import time
import door
import torch

Player = player.Player
Enemy = enemy.Enemy
AmmoPack = ammopack.AmmoPack
Door = door.Door

# Set up a timer to keep track of elapsed time
ammo_pack_timer = time.perf_counter()

# Initialize Pygame
pygame.init()

# Set up the Pygame window
screen_width, screen_height = 640, 480
screen = pygame.display.set_mode((screen_width, screen_height))

# Create player.py characters
player1 = Player(start_pos=(200, 200), color=(255, 0, 0), left_key=pygame.K_a, right_key=pygame.K_d, up_key=pygame.K_w,
                 down_key=pygame.K_s, screen_width=screen_width, screen_height=screen_height, shoot_key=pygame.K_SPACE, can_collect_ammo=False)
player2 = Player(start_pos=(300, 300), color=(0, 0, 255), left_key=pygame.K_LEFT, right_key=pygame.K_RIGHT, up_key=pygame.K_UP,
                 down_key=pygame.K_DOWN, screen_width=screen_width, screen_height=screen_height, can_collect_ammo=True)
players = [player1, player2]

# Create enemies
enemies = []

enemy = Enemy(screen_height=screen_height, screen_width=screen_width, enemies=enemies)

# Create a clock object to control the framerate
clock = pygame.time.Clock()

# List to store ammo packs
ammo_packs = []

last_spawn_time = pygame.time.get_ticks()

def spawn_ammo_pack(packs):
    global ammo_pack_timer

    # Generate random coordinates within the screen bounds
    x = random.randint(0, screen_width)
    y = random.randint(0, screen_height)

    # Don't spawn a new ammo pack if there are already two on the screen
    if len(packs) >= 2:
        return

    # Get the current time
    current_time = time.perf_counter()

    # Spawn a new ammo pack every 5 seconds
    if current_time - ammo_pack_timer > 5:
        ammo_packs.append(AmmoPack(x, y))
        ammo_pack_timer = current_time

# Create a door
door = Door(screen_height=screen_height, screen_width=screen_width)

def reset_game(did_succeed_level):
    # Reset game objects
    global player1, player2, players, enemies, enemy, ammo_packs, door

    player1 = Player(start_pos=(200, 200), color=(255, 0, 0), left_key=pygame.K_a, right_key=pygame.K_d,
                     up_key=pygame.K_w,
                     down_key=pygame.K_s, screen_width=screen_width, screen_height=screen_height,
                     shoot_key=pygame.K_SPACE, can_collect_ammo=False)
    player2 = Player(start_pos=(300, 300), color=(0, 0, 255), left_key=pygame.K_LEFT, right_key=pygame.K_RIGHT,
                     up_key=pygame.K_UP,
                     down_key=pygame.K_DOWN, screen_width=screen_width, screen_height=screen_height,
                     can_collect_ammo=True)
    players = [player1, player2]
    enemies = []
    enemy = Enemy(screen_height=screen_height, screen_width=screen_width, enemies=enemies)
    ammo_packs = []
    door = Door(screen_height=screen_height, screen_width=screen_width)

# Main game loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    # Update game state
    player1.update(ammo_packs, player2, clock)
    player2.update(ammo_packs, player1, clock)
    for enemy in enemies:
        enemy.update()

    # Check for collisions between player.py bullets and enemies
    for player in players:
        for bullet in player.bullets:
            for enemy in enemies:
                if bullet.collides_with(enemy):
                    player.bullets.remove(bullet)

    for player in players:
        player.draw(screen)
    # Check if it's time to spawn a new enemy
    current_time = pygame.time.get_ticks()
    if current_time - last_spawn_time > random.randint(1000, 1500) and len(enemies) < 10:
        enemies.append(Enemy(screen_height=screen_height, screen_width=screen_width, enemies=enemies))
        last_spawn_time = current_time

    for enemy in enemies:
        enemy.draw(screen)

    spawn_ammo_pack(ammo_packs)

    for ammo_pack in ammo_packs:
        ammo_pack.draw(screen)

    door.update(player1.bullets + player2.bullets)

    door.draw(screen)

    for enemy in enemies:
        if player1.is_colliding(enemy) or player2.is_colliding(enemy):
            # reset the game
            reset_game(did_succeed_level=False)
            break

    # Check if both players are touching the door
    if door.color == (0, 255, 0)\
            and player1.x > door.x - door.size\
            and player1.x < door.x + door.size\
            and player1.y > door.y - door.size\
            and player1.y < door.y + door.size\
            and player2.x > door.x - door.size\
            and player2.x < door.x + door.size\
            and player2.y > door.y - door.size\
            and player2.y < door.y + door.size:
        # Reset game objects
        reset_game(did_succeed_level=True)

    # Initialize the font
    font = pygame.font.Font(None, 36)

    # Create a text surface and rect to hold the text
    text_surface = font.render(f"Ammo: {player1.ammo}", True, (0, 0, 0))
    text_rect = text_surface.get_rect()

    # Set the text rect to the top left corner of the screen
    text_rect.topleft = (10, 10)

    # Draw the text to the screen
    screen.blit(text_surface, text_rect)

    pygame.display.flip()

    # Limit the framerate to 60 fps
    clock.tick(60)

    # Draw the game to the screen
    screen.fill((255, 255, 255))  # Fill the screen with white
