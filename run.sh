rm main
g++ --std=c++11 HunIPU.cpp -lpoplar -lpopops -lpoputil -lpoplin -o main
./main 4096
