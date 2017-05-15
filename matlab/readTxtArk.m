function x = readTxtArk(filename)
%Syntax: x = readTxtArk(filename)
%read dataitema from Kaldi ark txt file
%return a struct vector of #uterances length with fields name and data

s = textread(filename, '%s', 'delimiter', '', 'bufsize', 8*1024*1024);
start = 1;
x = [];
for k = 1:length(s);
    if start == 1 %name line
        ss = 'x(end).data = [';
        start = 2;
        x(end+1).name = strtrim(s{k}(1:end-2));
        continue
    end
    if start == 2
        ss = [ss s{k}];
        start = 0;
    else
        ss = [ss ';' s{k}];
    end
    if any(s{k}(end-1:end) == ']')
        eval([ss ';']);
        start = 1;
    end
end

