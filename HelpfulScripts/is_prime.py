def is_prime(num:int):
    assert num > 0 and type(num) == int
    if num == 1: return False
    if num == 2: return True
    for i in range(2, int(num / 2) + 1):
        if num % i == 0:
            print(f'no, it is divisible by {i}')
            return False
    print(f'{num} is prime')
    return True

def main(inp):
    is_prime(inp)

if __name__ == '__main__':
    inp = int(input('What number should we check?\n'))
    main(inp)