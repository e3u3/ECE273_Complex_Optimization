function [ K ] = rbf_kernel( U, V, gamma )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[m, ~] = size(U);
p = size(V, 1);
K = zeros(m,p);tic;
for i = 1:m
    
        tmp = V-U(i,:);
        K(i,:) = diag(tmp*tmp')';
    
end
toc;
K = exp(-1.*gamma.*K);
end

