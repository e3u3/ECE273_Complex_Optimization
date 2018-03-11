clear; close all; clc;

%% load both training and testing data, as well as preprepared kernel results
load 'data.mat';
load 'rbf.mat';

%% some parameters
C = 2;
gamma = 2^-7;
dc_0=[];
compute_kernel_agian = 0;

%% training 10 binary classifiers
for cn = 9:9
%% design labels for one-vs-all
train_lbl = label_proc(train_label, cn);
test_lbl = label_proc(test_label, cn);
X = train_img;
y = train_lbl;

%% training
model = rbf_svm( X, y, C, gamma, compute_kernel_agian );

pos = model.pos;
alpha = model.alpha;
b = model.b;

%% compute testing error
% use pre-compute kernel to save time, of course, you can uncomment them to compute agian 
% K2 = rbf_kernel( X, test_img, gamma );
test = sign(test_lbl'.*(sum(K2(pos,:).*alpha,1)-b));
dc = sum(K2(pos,:).*alpha,1)-b;
dc_0 = [dc_0,dc'];
error2 = sum(test==-1)/1000;
disp(1-error2);

end

%% compute the total error for multiclass classification
[~,result] = max(dc_0,[],2);
result = result-1;
error = sum(result~=test_label)/1000;
