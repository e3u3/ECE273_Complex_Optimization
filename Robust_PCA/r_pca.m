clear; close all; clc;

is_old_data = 0;

if ~is_old_data
    dir0 = '.\CroppedYale\yaleB24';
    cd CroppedYale\yaleB24;
    files = dir('*.pgm');
    cd ../..;
    [n_fig,~] = size(files);
    D = [];
    for i=1:n_fig-1
        fname = [dir0, '\',files(i).name];
        im = pgmRead( fname );
        im_t = reshape(im,[],1);
        D = [D,im_t];
    end
    
    save('data24.mat','D');
else
    load data.mat
end

[m,~] = size(D);

lambda = m^-0.5;

A_c = zeros(size(D));
A_p = A_c;
E_c = A_c;
E_p = A_c;

t_c = 1;
t_p = 1;
mu_c = norm(D,2);
mu_bar = 1e-5*mu_c;
times = 0;
while 1
    times = times + 1;
    
    A_sig_c = A_c + (t_p-1)/t_c .* (A_c - A_p);
    E_sig_c = E_c + (t_p-1)/t_c .* (E_c - E_p);
    
    Y_A_c = A_sig_c - 0.5.*(A_sig_c + E_sig_c - D);
    
    A_p = A_c;
    E_p = E_c;
    
    [U, S, V] = svd(Y_A_c,'econ');
    S_temp = S - mu_c/2.*eye(size(S));
    A_c = U*(S_temp.*double(S_temp>0))*V';
    
    Y_E_c = E_sig_c - 0.5.*(A_sig_c + E_sig_c - D);
    Y_temp = abs(Y_E_c)-lambda*mu_c/2;
    E_c = sign(Y_E_c).*(Y_temp.*double(Y_temp>0));
    t_p = t_c;
    t_c = (1+sqrt(1+4*t_p^2))/2;
    mu_c = max(0.9*mu_c, mu_bar);
    
    %     flag0 = norm(A_c-A_p,2);
    %     flag1 = norm(E_c-E_p,2);
    %     if(flag0+flag1<0.01)
    %         break;
    %     else
    if(times>100)
        break;
    end
end
disp(times);

tid = 46;
train = reshape(D(:,tid),192,168);
test = reshape(A_c(:,tid),192,168);
testE = reshape(E_c(:,tid),192,168);
figure;
subplot(1,3,1);
imshow(train,[]);
subplot(1,3,2);
imshow(test,[]);
subplot(1,3,3);
imshow(testE,[]);