% Simple Random Forest Toolbox for Matlab
% written by Mang Shao and Tae-Kyun Kim, June 20, 2014.
% updated by Tae-Kyun Kim, Feb 09, 2017

% This is a guideline script of simple-RF toolbox.
% The codes are made for educational purposes only.
% Some parts are inspired by Karpathy's RF Toolbox

% Under BSD Licence

% Initialisation
init;

% Select dataset
[data_train, data_test] = getData('Toy_Spiral'); % {'Toy_Gaussian', 'Toy_Spiral', 'Toy_Circle', 'Caltech'}
plot_toydata(data_train)
figure;
%%%%%%%%%%%%%
% check the training and testing data
    % data_train(:,1:2) : [num_data x dim] Training 2D vectors
    % data_train(:,3) : [num_data x 1] Labels of training data, {1,2,3}


%4 training bags, size half of original dataset
bag_size=uint8(size(data_train,1)/2);
training_bags=zeros(4,bag_size, 3);

for bag= 1:4
    rand_select=randperm(size(data_train,1));
    training_bags(bag,:,:)= data_train(rand_select(1, 1:bag_size),:);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%TODO
%Try diff bagging techniques----------------------------------------------------
%Try diff splitting functions (singular dimension, more random, more
%diverse trees, weaker, less correlation)-----------------------------------------

%How join trees leaf prob distrs, product, avg? how robust to
%noise?majority voting?-------------------------------------------------------------------done
%construction/prediction depends Model params: tree dept, randomness,info gain smaller that predefined value, avoiding growing full trees improves generalisation
ours has max tree depth, and emptiness stop criteria (5)-------------------------------

%(controlled by p)and its type, type. choice weak learner model, training
%obj function. Affect confidence, efficiency(COMPARE TIMES),  accuracy
%deep trees can ->overfitting. Test accuracy increase with forest size.
%how randomness affects tree correlation & generalisation
%choice of stopiing criteria, influence tree shape, whether balanced, (want
%to avoid un), at limit may become chain of weak learners, little feature
%sharing, thus little generalisation.
%LOOK AT NOTES SUMMARY FOR INTRO/CONCLUSION
%Curremtly use axis aligned weak learner, pick feature randomly, pick rand
%val in range as threshold of chosen feature, repeat the splitNum times,
%choose best split. DISPLAY SPLITS & CHOICE?
%Use fixed depth, emptiness as stop criteria.
%show 4 test points on graph with orig swirl with colours? ------------------------------------done
%no trees (1,3,5,10,20)with 5 3 'IG'
%DEPTH [2,5,7,11] fix 20,3,IG

Randomised Forests is an ensemble of bagged tree learners with randomized feature selection.
Variants usually modify one or several of the following components
of RF:
–
Node objective function
I, a criterion for selecting the best node weak learner (split feature, threshold)
–
Split feature function,
𝜙𝜙(v)·𝜓𝜓,for splitting data v.
–
Predictor, a function that is used for predicting output
y.
–
Tree structure, which inherently has a hierarchical architecture.
Each node has associated a different test function.
–
We formulate a test function at a split node
j as a function with binary outputs
h(v,𝜃𝑗) ={0 or1}
–
𝜃𝑗 ∈ denote the parameters of the test function at the j-thsplit node.
–
The data point
v arriving at the split node is sent to its left or right child node according to the result of the test function.
Different types of split functions have been extensively discussed: linear,
non-linear, axis-aligned, and two-pixel test.

The filter function
𝜙 selects some features of choice out of the entire vector v.
–
𝜓 defines the geometric primitive used to separate the data (e.g., an axis-aligned hyperplane, an oblique hyperplane, a general surface etc.)
–
The parameter 𝜏 is a threshold for the inequalities used in the binary test.

parametrization of the weak learner model as
θ=(𝜙,𝜓,𝜏). Learn function based on incoming data subset that best splits the data. maximisation objective function to split node data, max info gain

result of the optimization problem determines the parameters of the
weak learners, which in turn determines the path followed by a data and its prediction.

Thus, if we use the children (rather than the parent node) we would have
more chances of correct prediction; we have reduced the uncertaintyof prediction.
–
This intuitive explanation can be formulated using quantitative measures for
entropyand information gain.

For large dimensional problems would choose random subset of param vals for efficiency, ours only 2d,
control randomness through size subset to size set randomeness
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    % data_test(:,1:2) : [num_data x dim] Testing 2D vectors, 2D points in the
    % uniform dense grid within the range of [-1.5, 1.5]
    % data_train(:,3) : N/A
    
scatter(data_test(:,1),data_test(:,2),'.b');
figure;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Que 1 part 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%{
% Set the random forest parameters for instance, 
param.num = 1;         % Number of trees 10
param.depth = 2;        % trees depth 5 
param.splitNum = 3;     % Number of split functions to try
param.split = 'IG';     % Currently support 'information gain' only

%%%%%%%%%%%%%%%%%%%%%%
% Train Random Forest 

%tree = growTrees(training_bags(1,:, :),param);

% Grow all trees
trees = growTrees(data_train,param);
%l_node=trees(1).node(2);
%r_node=trees(1).node(3);
%l_node_data=data_train(l_node.idx.', :);
%fig =visualise(l_node_data,l_node.prob,[],false) %doesnt work as l_node.prob is not 3D
%get(fig)
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Que 1 - part b
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%{
param.num = 1;         % Number of trees 10
param.depth = 10;        % trees depth 5 
param.splitNum = 3;     % Number of split functions to try
param.split = 'IG';     % Currently support 'information gain' only
tic
tree = growTrees(data_train,param);
toc
%9 example leaf renders
rand_leafs=randperm(size(tree.leaf,2));
no_div=9;
for L= 1:no_div %length(tree.leaf)
    subplot(3,3,L);
    bar(tree(1).leaf(rand_leafs(L)).prob);
    axis([0.5 3.5 0 1]);
end
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Que 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.num = 10;         % Number of trees 10
param.depth = 5;        % trees depth 5 
param.splitNum = 3;     % Number of split functions to try
param.split = 'IG';     % Currently support 'information gain' only
tic
trees = growTrees(data_train,param);
toc
%{
test_point = [-.5 -.7; .4 .3; -.7 .4; .5 -.5];
for n=1:4
    %retrieves indexes of leaves that test_point ends at
    leaves = testTrees(test_point(n,:),trees);
    %leaves are 10 tree outputs for each test point
    %=tree(T).leaf(leaf_idx).label= cntTotal = index in tree(1).prob(cnt_total,:) = tree(T).node(n).prob';
    
    % average the class distributions of leaf nodes of all trees
    p_rf = trees(1).prob(leaves,:); %10 trees in forest, 10 leaf distributions.
    p_rf_sum = sum(p_rf)/length(trees);
    figure;
    title(sprintf('Class distribution of dest leaf in each decision tree & avg, datapoints %A.',test_point(n,:)));
    for L= 1: size(p_rf,1) %length(tree.leaf)
        subplot(4,3,L);
        bar(p_rf(L));
        axis([0.5 3.5 0 1]);
        title(sprintf('Leaf %i',L));
    end
    subplot(4,3,(size(p_rf,1)+1))
    bar(p_rf_sum);
    title(sprintf('Average'));
end
%}
% Test on the dense 2D grid data, and visualise the results ... 

    
    
    % average the class distributions of leaf nodes of all trees
    %data_test length by 10 trees, average, to data_test length *3
    %select largest class, then plot
    %choose voting scheme
    voting_scheme=["sum-avg", "maj-vote", "product"]
    for vote =1:length(voting_scheme)
        tic
        for m=1:size(data_test,1)
            leaves = testTrees(data_test(m,1:2),trees);
            p_rf = trees(1).prob(leaves,:); %10 trees in forest, 10 leaf distributions.
            if (voting_scheme(vote)== "sum-avg")
                p_rf_sum = sum(p_rf)/length(trees);
                [val, idx] = max(p_rf_sum);
                data_test(m,3)=idx;
            end
            if (voting_scheme(vote)== "product")
                p_rf_sum = times(p_rf)/length(trees);
                [val, idx] = max(p_rf_sum);
                data_test(m,3)=idx;
            end
            if (voting_scheme(vote)== "maj-vote")
                total=[0,0,0];
                for i = 1:size(p_rf,1)
                    [val,idx]=max(p_rf(i,:));
                    total(idx)+=1;
                end
                [val, idx] = max(total);
                data_test(m,3)=idx;
            end
        end
        toc
        plot_toydata(data_test);
    end
    
    





% Change the RF parameter values and evaluate ...


%for i=1: 2^param.depth-1
    %if is leaf node, not 0 data, but no split
%    if ~tree(1).node(i).dim && ~isempty(tree(T).node(n).idx)
        
%    end
%end

%%%%%%%%%%%%%%%%%%%%%
% Evaluate/Test Random Forest

% grab the few data points and evaluate them one by one by the leant RF
test_point = [-.5 -.7; .4 .3; -.7 .4; .5 -.5];
dataAll=data_train;
for n=1:4
    leaves = testTrees([test_point(n,:) 0],trees);
    % average the class distributions of leaf nodes of all trees
    p_rf = trees(1).prob(leaves,:);
    p_rf_sum = sum(p_rf)/length(trees);
    [val, idx] = max(p_rf_sum);
    dataAll=[dataAll; test_point(n,1:2),idx];
end
plot_toydata(data_test);


% Test on the dense 2D grid data, and visualise the results ... 

% Change the RF parameter values and evaluate ... 



%{

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% experiment with Caltech101 dataset for image categorisation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

init;

% Select dataset
% we do bag-of-words technique to convert images to vectors (histogram of codewords)
% Set 'showImg' in getData.m to 0 to stop displaying training and testing images and their feature vectors
[data_train, data_test] = getData('Caltech');
close all;



% Set the random forest parameters ...
% Train Random Forest ...
% Evaluate/Test Random Forest ...
% show accuracy and confusion matrix ...


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% random forest codebook for Caltech101 image categorisation
% .....
%}



