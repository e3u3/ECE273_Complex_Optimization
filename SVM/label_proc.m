function output = label_proc(input, no)
output = input;
output(input==no) = 1;
output(input~=no) = -1;
end