import random

def get_subset(arr, size=0.5):
    r = random.Random(42)
    r.shuffle(arr)
    return arr[:int(len(arr) * size)]