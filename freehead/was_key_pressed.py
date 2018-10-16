import pygame


def was_key_pressed(key):
    keydowns = []
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            keydowns.append(event.key)
    return key in keydowns
