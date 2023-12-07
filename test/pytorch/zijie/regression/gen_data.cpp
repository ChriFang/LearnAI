#include <iostream>

float w = 6.666;
float b = 1.111;

const int size = 100;
int main()
{
  float a[size];
  srand((unsigned)time(NULL));
  for (int i = 0; i < size; i++)
  {
    int r_num = rand();
    int randomInt = r_num % 100;
    float randomFloat = static_cast<float>(r_num) / RAND_MAX;
    float x = static_cast<float>(randomInt + randomFloat);
    double y = w * x + b;
    std::cout << x << ", " << y << std::endl;
  }
}
