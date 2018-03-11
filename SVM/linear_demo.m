clear; close all; clc;

%% load training and testing data
load 'data.mat';

%% some parameters
C = 2;
model_0=[];
dc_0 = [];
error_0 = [];

%% training 10 binary classifiers
for id=0:9
    disp(['__________',num2str(id),'___________']);
	%% design one-vs-all labels
    train_lbl = label_proc(train_label, id);
    test_lbl = label_proc(test_label, id);
    tic;
	%% training binary SVM
    [ model ] = linear_svm( train_img, train_lbl, C );
    toc;
    model_0 = [model_0 model];
	%% compute testing error
    dc = ((model.w'*test_img')-model.b)';
    dc_0 = [dc_0, dc];
    test = sign(test_lbl'.*((model.w'*test_img')-model.b))';
    error = 1-sum(test==-1)/1000;
    error_0 = [error_0 error];
end

%% compute error of multiclass classification
[~,result] = max(dc_0,[],2);
result = result-1;
error = sum(result~=test_label)/1000;

%% compute norm of w
for id=1:10
    temp = model_0(id).w;
    norm_w(id) = sqrt(temp'*temp);
end
norm_w = norm_w';