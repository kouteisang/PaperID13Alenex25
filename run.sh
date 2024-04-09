rm main
g++ --std=c++11 HunIPU.cpp -lpoplar -lpopops -lpoputil -lpoplin -o main
./main 1024 test_data/sparse/new-sparse_1024.txt