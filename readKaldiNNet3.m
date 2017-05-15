function nn = readKaldiNNet3(filename)
%Syntax: nn = readKaldiNNet3(filename)
%txt format supported only
%return a structure with following cell vectors:
%	.input_nodes
%	.component_nodes
%	.output_nodes
%	.components

%header part:
S = textread(filename, '%s', 'delimiter', '', 'bufsize', 8*1024*1024);
if isempty(S) || ~strcmp(S{1}, '<Nnet3> ')
    error('Wrong format - must be the Kaldi Nnet3 txt format')
end

n = 2;
nn.input_nodes = {};
nn.component_nodes = {};
nn.output_nodes = {};
nn.components = {};
while ~isempty(S{n}) && ~strcmp(S{n}(1:16), '<NumComponents> ')
   s1 = strsplit(S{n});
   switch s1{1}
      case 'input-node'
          nn.input_nodes{end+1} = parseInputNode(S{n});
      case 'component-node'
          nn.component_nodes{end+1} = parseComponentNode(S{n});
      case 'output-node'
          nn.output_nodes{end+1} = parseOutputNode(S{n});
      otherwise
         warning(['Unrecognized node "' s1{1} '"'])
   end
   n = n + 1;
end
if isempty(S{n})
    n = n + 1;
end

%<NumComponents> 25 
if length(S) < n || isempty(S{n}) || ~strcmp(S{n}(1:16), '<NumComponents> ')
    error('Wrong format - <NumComponents> missing - must be the Kaldi Nnet3 txt format')
end
NC = str2double(S{n}(17:end));
n = n + 1;
for k = 1:NC
    [nn.components{end+1}, shift] = parseComponent(S(n:end));
    n = n + shift;
end

function node = parseInputNode(s)
%parse node line and return a structure
f = find(s == ' ' | s == '=');
node.name = s(f(2)+1:f(3)-1);
node.dim = str2double(s(f(4)+1:end));

function node = parseComponentNode(s)
%parse node line and return a structure
f = find(s == ' ' | s == '=');
node.name = s(f(2)+1:f(3)-1);
node.component = s(f(4)+1:f(5)-1);
node.input = s(f(6)+1:end);

function node = parseOutputNode(s)
%parse node line and return a structure
f = find(s == ' ' | s == '=');
node.name = s(f(2)+1:f(3)-1);
node.input = s(f(4)+1:f(5)-1);
node.objective = s(f(6)+1:end);

function [node, shift] = parseComponent(S)
%parse component and return a structure and number of "consumed" lines
if ~strcmp(S{1}(1:15), '<ComponentName>')
    error('Wrong format - <ComponentName> missing - must be the Kaldi Nnet3 txt format')
end
shift = 1;
s1 = strsplit(S{1}(17:end));
node.ComponentName = s1{1};
node.ComponentType = s1{2}(2:end-1);

switch node.ComponentType
    case 'FixedAffineComponent'
        n = 2;
        while S{n}(1) ~= '<'
            n = n + 1;
        end
        dim = length(str2num(S{2}));
        node.LinearParams = zeros(n-2, dim);
        for k = 2:n-2
            node.LinearParams(k-1, :) = str2num(S{k});
        end
        node.LinearParams(end, :) = str2num(S{n-1}(1:end-1));
        if ~strcmp(S{n}(1:12), '<BiasParams>')
            error('Wrong format - <BiasParams> missing in <FixedAffineComponent> - must be the Kaldi Nnet3 txt format')
        end
        node.BiasParams = str2num(S{n}(16:end-1));
        shift = n + 1;
        
    case 'NaturalGradientAffineComponent' %identical to FixedAffineComponent
        n = 2;
        while S{n}(1) ~= '<'
            n = n + 1;
        end
        dim = length(str2num(S{2}));
        node.LinearParams = zeros(n-2, dim);
        for k = 2:n-2
            node.LinearParams(k-1, :) = str2num(S{k});
        end
        node.LinearParams(end, :) = str2num(S{n-1}(1:end-1));
        if ~strcmp(S{n}(1:12), '<BiasParams>')
            error('Wrong format - <BiasParams> missing in <FixedAffineComponent> - must be the Kaldi Nnet3 txt format')
        end
        node.BiasParams = str2num(S{n}(16:end-1));
        shift = n + 1; %one extra line
        
    case 'PnormComponent'
        node.InputDim = str2double(s1{4});
        node.OutputDim = str2double(s1{6});
        
    case 'NormalizeComponent'
        node.InputDim = str2double(s1{4});
        node.TargetRms = str2double(s1{6});
        node.AddLogStddev = strcmp(s1{8}, 'T');
        
    case 'FixedScaleComponent'
        f = find(S{1} == '[');
        node.Scales = str2num(S{1}(f+1:end-1));
        shift = 2;
        
    case 'LogSoftmaxComponent'
        node.Dim = str2double(s1{4});
        shift = 3;
        
    otherwise
        error(['Unsupported component type "' node.ComponentType '"'])
end

