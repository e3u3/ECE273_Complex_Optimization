function [ model ] = rbf_svm( X, y, C, gamma, compute_kernel_agian )

%% number of training samples
n = length(y);

%% calculate kernel however this does takes time! 2000*2000 needs about 6 minites
% I'm sorry for not finding a fast way to compute this
if compute_kernel_agian == 0
    load 'rbf.mat';
else
    K = rbf_kernel( X, X, gamma );
end

%% solve this by using cvx toolbox
cvx_begin
    variable a(n)
    minimize( 0.5.*(y.*a)'*K*(y.*a) - ones(1,n)*a )
    subject to
        0 <= a <= C;
        y'*a == 0;
cvx_end

%% find SV
pos = a>1e-6;
alpha = a(pos,:).*y(pos,:);
K_X = K(pos,:);

%% calculate bias term
pos2 = a<1.999;
pos3 = pos&pos2;
K_X_ = K(:,pos);
b = mean(sum(K_X_(pos3,:).*alpha',2));

%% compute training error
train = sign(y'.*(sum(K_X.*alpha,1)-b))';
error = sum(train==-1)/n;
disp(1-error);

%% save the model
model.pos = pos;
model.b = b;
model.a = a;
model.alpha = alpha;
end

