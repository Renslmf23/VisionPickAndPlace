import random

blue = (0, 0, 255)
green = (0, 255, 0)
red = (255, 0, 0)
pink = (255, 0, 255)
cyan = (0, 255, 255)
yellow = (255, 255, 0)
black = (0, 0, 0)
white = (255, 255, 255)

def get_random_color():
    return tuple((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))