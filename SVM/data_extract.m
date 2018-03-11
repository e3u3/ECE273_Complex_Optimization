clear; close all; clc;

imgFile = 'train-images.idx3-ubyte';
labelFile = 'train-labels.idx1-ubyte';
readDigits = 2000;
imgFile2 = 't10k-images.idx3-ubyte';
labelFile2 = 't10k-labels.idx1-ubyte';
readDigits2 = 1000;
offset = 0;
[train_img, train_label] = readMNIST(imgFile, labelFile, readDigits, offset);
[test_img, test_label] = readMNIST(imgFile2, labelFile2, readDigits2, offset);

save('data.mat','train_img','train_label','test_img','test_label');