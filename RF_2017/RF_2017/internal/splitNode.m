function [node,nodeL,nodeR] = splitNode(data,node,param)
% Split node

visualise = 0;
%splitfunct="randrange";%other split functions

% Initilise child nodes
iter = param.splitNum;
nodeL = struct('idx',[],'t',nan,'dim',0,'prob',[]);
nodeR = struct('idx',[],'t',nan,'dim',0,'prob',[]);

if length(node.idx) <= 5 % make this node a leaf if has less than 5 data points
    node.t = nan;
    node.dim = 0;
    return;
end

idx = node.idx;
data = data(idx,:);
[N,D] = size(data);
ig_best = -inf; % Initialise best information gain
idx_best = [];
for n = 1:iter
    t=0;
    dim=-1;
    idx_=[];
    % Split function - Modify here and try other types of split function
    if strcmp(param.split,'IG')
        dim = randi(D-1); % Pick one random dimension
        d_min = single(min(data(:,dim))) + eps; % Find the data range of this dimension
        d_max = single(max(data(:,dim))) - eps;
        t = d_min + rand*((d_max-d_min)); % Pick a random value within the range as threshold
        idx_ = data(:,dim) < t;
    else
        if strcmp(param.split,'twopixel')
            dim=12;
            d1_min = single(min(data(:,1))) - eps; % Find the data range of this dimension
            d1_max = single(max(data(:,1))) + eps;
            d2_min = single(min(data(:,2))) + eps; % Find the data range of this dimension
            d2_max = single(max(data(:,2))) - eps;
            t = (d1_min + rand*((d1_max-d1_min))) - (d2_min + rand*((d2_max-d2_min))); % Pick a random value within the range as threshold
            idx_ = (data(:,1)-data(:,2)) < t;
        end
    end
    %{
    if strcmp(param.split,'kmean')
        dim=4;
        idx = kmeans(data,2);%cluster indes for each pt
        idx_= find(idx==1); % data(find(idx==1),:);
    end
    %}
    %idx_
    ig = getIG(data,idx_); % Calculate information gain
    
    if visualise
        visualise_splitfunc(idx_,data,dim,t,ig,n);
        pause();
    end
    
    [node, ig_best, idx_best] = updateIG(node,ig_best,ig,t,idx_,dim,idx_best);
    
end
%tree(T).leaf(leaf_idx).label; to
 %if ~isempty(tree(T).node(n).idx)
  %   if ~tree(T).node(n).dim % if this is a leaf node dim=0
   %      tree(T).leaf(cnt).label = cnt_total;
   
   %idx is logical array 0's, 1's
nodeL.idx = idx(idx_best);
nodeR.idx = idx(~idx_best);

if visualise
    visualise_splitfunc(idx_best,data,dim,t,ig_best,0)
    fprintf('Information gain = %f. \n',ig_best);
    pause();
end

end

function ig = getIG(data,idx) % Information Gain - the 'purity' of data labels in both child nodes after split. The higher the purer.
L = data(idx);
R = data(~idx);
H = getE(data);
HL = getE(L);
HR = getE(R);
ig = H - sum(idx)/length(idx)*HL - sum(~idx)/length(idx)*HR;
end

function H = getE(X) % Entropy
cdist= histc(X(:,1:end), unique(X(:,end))) + 1;
cdist= cdist/sum(cdist);
cdist= cdist .* log(cdist);
H = -sum(cdist);
end

function [node, ig_best, idx_best] = updateIG(node,ig_best,ig,t,idx,dim,idx_best) % Update information gain
if ig > ig_best
    ig_best = ig;
    node.t = t;
    node.dim = dim;
    idx_best = idx;
else
    idx_best = idx_best;
end
end