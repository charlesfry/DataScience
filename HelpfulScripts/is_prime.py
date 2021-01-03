import math

def is_prime(num:int):
    assert num > 0 and type(num) == int
    if num == 1: return False
    if num == 2: return True
    for i in range(2, int(num / 2) + 1):
        if num % i == 0:
            print(f'no nerd, it is divisible by {i}')
            return False
    print(f'{num} is prime')
    return True

def main():
    print(is_prime(121332951))

if __name__ == '__main__':

    inp = input('What number should we check?')
    print(is_prime(int(inp)))