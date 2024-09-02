# Title: Linear Assignment on Tile-Centric Accelerators: Redesigning Hungarian Algorithm on IPUs

This is the readme file for the Paper "Linear Assignment on Tile-Centric Accelerators: Redesigning Hungarian Algorithm on IPUs".

## Authors:

Cheng Huang, Alexander Mathiasen, Josef Dean, Johannes Langguth, Davide Mottin, Ira Assent.

## Hardware Platform:

IPU: We run our algorithm on the 1.325GHz Mk2 GC200 IPU.

GPU: All the other algorithms run on an Nvidia A100 GPU with 40GB memory.

## Datasets:

In this repo, we provide the datasets, including 

(1) The sparse dataset with a matrix size of 1024 (test_data/sparse/new-sparse_1024.txt) 

(2) The real-world datasets for graph alignment. (real-world/).

(3) We put the remaining datasets on google drive due to the space limited, and can be access from https://drive.google.com/drive/folders/1It5Uq9_u6Gvft41BQRKglFrjArH_h21s?usp=sharing

## Running script

We include the run.sh file to run our algorithm, in detail, the run.sh including the following command.

```
rm main 
g++ --std=c++11 HunIPU.cpp -lpoplar -lpopops -lpoputil -lpoplin -o main
./main 1024 test_data/sparse/new-sparse_1024.txt
```

The first line means we remove the previously compiled program (if any).

The second line means we compile the program.

The third line is to actually run the program. The third line is in the following format.

```
./program matrix-size data-source
```

For example, after we compile the program and generate main, and we want to test the matrix size with 1024, the data is in test_data/sparse/new-sparse_1024.txt. We can run the following command.

```
./main 1024 test_data/sparse/new-sparse_1024.txt
```

After running the algorithm, the program will output the running time.

## Baseline: 

We compare our algorithm with the following baseline algorithms. We list the GitHub repo in the following.

1. FastHA: https://github.com/paclopes/HungarianGPU

2. CuLAP: https://github.com/tianluoabc/CuLAP

## Contact:

For any questions, please feel free to contact cheng [AT] cs [DOT] au [AOT] dk
