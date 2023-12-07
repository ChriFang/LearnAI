import random

w = 6.666
b = 1.111

def generate_random_float():
    # 生成一个介于0和100之间的随机浮点数
    return random.random() * 100

if __name__ == '__main__':
  count = 0
  while count < 100:
    x = generate_random_float()
    print("{0}, {1}".format(x, w * x + b))
    count += 1
