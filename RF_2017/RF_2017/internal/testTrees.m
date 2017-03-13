function label = testTrees(data,tree)
% Slow version - pass data point one-by-one

cc = [];
for T = 1:length(tree)%number of trees
    for m = 1:size(data,1);%amount of test data
        idx = 1;
        
        while tree(T).node(idx).dim
            t = tree(T).node(idx).t;
            dim = tree(T).node(idx).dim;
            % Decision
            %if axis alligned
            if dim==12
                if (data(m,1)-data(m,2)) < t % Pass data to left node
                    idx = idx*2;
                else
                    idx = idx*2+1; % and to right
                end
            %if two pixel test
            else
                 if data(m,dim) < t % Pass data to left node
                    idx = idx*2;
                else
                    idx = idx*2+1; % and to right
                end               
            end
            
        end
        %idx is leaf node
        leaf_idx = tree(T).node(idx).leaf_idx;
        
        if ~isempty(tree(T).leaf(leaf_idx))
            p(m,:,T) = tree(T).leaf(leaf_idx).prob;%distribution at leaf
            if(tree(T).leaf(leaf_idx).label==0)
                printfn ('leaf_idx in tree(T).leaf that gives 0 label: %A',leaf_idx);
            end
            label(m,T) = tree(T).leaf(leaf_idx).label;
            
%             if isfield(tree(T).leaf(leaf_idx),'cc') % for clustering forest
%                 cc(m,:,T) = tree(T).leaf(leaf_idx).cc;
%             end
        end
    end
end

end

