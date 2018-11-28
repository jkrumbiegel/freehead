import pygame


def was_key_pressed(*keys):
    keydowns = [e.key for e in pygame.event.get() if e.type == pygame.KEYDOWN]

    return tuple(key in keydowns for key in keys) if len(keys) > 1 else keys[0] in keydowns
