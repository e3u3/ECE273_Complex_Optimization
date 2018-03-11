function [ model ] = linear_svm( X, y, C )

%% number of training samples
n = length(y);

%% linear kernel, lol
K = X*X';

%% using cvx toolbox
cvx_begin
    variable a(n)
    minimize( 0.5.*(y.*a)'*K*(y.*a) - ones(1,n)*a )
    %minimize( 0.5.*(a)'*K*(a) - ones(1,n)*a )
    subject to
        0 <= a <= C;
        y'*a == 0;
cvx_end

%% find SV and compute w and bias
pos = a>1e-6;
SV = X(pos,:);
m = size(SV,1);
alpha = a(pos,:).*y(pos,:);
w = SV'*alpha;
b = (mean(w'*SV(alpha>0,:)') + mean(w'*SV(alpha<0,:)'))/2;                                                                                                                                                                                                           

%% compute training error
train = sign(y'.*((w'*X')-b))';
error = sum(train==-1)/n;
disp(error);

%% save model
model.SVs = SV;
model.w = w;
model.b = b;
model.a = a;
model.nSV = m;
model.alpha = alpha;
end

