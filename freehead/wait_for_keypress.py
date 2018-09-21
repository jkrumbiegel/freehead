import pygame


def wait_for_keypress(*keys):
    # flush events first so old keypresses don't count
    pygame.event.clear()
    key_detected = False
    while not key_detected:
        keydowns = []
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                keydowns.append(event.key)
        keys_pressed = [key in keydowns for key in keys]
        if sum(keys_pressed) != 1:
            # only allow single wanted keypress at a time
            continue
        else:
            for (is_pressed, key) in zip(keys_pressed, keys):
                if is_pressed:
                    return key
        time.sleep(0.001)