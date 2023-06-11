clc;
clear all;
close all;
format short;

%----------------------------- reading train images
list_files = dir('train');

for i= 1:numel(list_files)
    f = list_files(i).name;
    if length(f)>3
        images_train{i-2,1} =  f(1:end-4) ;
        images_train{i-2,2} = readImageRaw(append('train/',f), 128, 128)./255.0;
    end
end

%----------------------------- reading test images
list_files = dir('test');

for i= 4:15
    f = list_files(i).name;
    
    images_test{i-3,1} = str2num( f(1:end-4) );
    images_test{i-3,2} = readImageRaw(append('test/',f), 128, 128)./255.0;  
end
images_test = sortrows(images_test, [1], {'ascend'});

% read test_label 
fileID = fopen('test/test_label.txt','r');
test_label = textscan(fileID,'%d %s');
fclose(fileID);
test_label = test_label{1,2};


%----------------------------- Law's filters
w{1,1} = 'L5';   w{1,2} = [1, 4, 6, 4, 1]';     %L5  
w{2,1} = 'E5';   w{2,2} = [-1, -2, 0, 2, 1]';   %E5  
w{3,1} = 'S5';   w{3,2} = [-1, 0, 2, 0, -1]';   %S5  
w{4,1} = 'W5';   w{4,2} = [-1, 2, 0, -2, 1]';   %W5
w{5,1} = 'R5';   w{5,2} = [1, -4, 6, -4, 1]';   %R5

filters = cell(25,2);
for i = 1:5
    for j = 1:5
        filters{5*(i-1)+j,1} = append(w{i,1},w{j,1});
        filters{5*(i-1)+j,2} = kron(w{i,2},w{j,2}');
        
    end
end

% normilze_L5L5 = sum( filters{1,2}, 'all');
% filters{1,2} = filters{1,2}./normilze_L5L5;

%------------------ Filter response & Energy feature averaging (train)
feature_response = cell(36,25);
energy_responses = zeros(36,25);

for i =1:36
    for j= 1:25
        A = padarray(images_train{i,2},[2 2],'symmetric');
        feature_response{i,j} = conv2(A,filters{j,2},'valid');
        energy_responses(i,j) = mean( feature_response{i,j}.^2 , 'all');

    end
end

% adjusting the r1 value (L5L5) by substracting the mean
% mean_r1 = mean( energy_responses(:,1), 1);
% fff = energy_responses(:,1) - mean_r1;

%------------------ Filter response & Energy feature averaging (test)
feature_response_test = cell(12,25);
energy_resp_test = zeros(12,25);

for i =1:12
    for j= 1:25
        A = padarray(images_test{i,2},[2 2],'symmetric');
        feature_response_test{i,j} = conv2(A,filters{j,2},'valid');
        energy_resp_test(i,j)= mean(feature_response_test{i,j}.^2 ,'all');
    end
end  

%----------------------------- the strongest discriminant power (feature)

% means for each class per each feature 25-D
j=1;
for i =1:4
    y_dot{i,1} = append('y_dot_class_', string(i) );
    y_dot{i,2} = sum(energy_responses(j:j+8,:), 1) / 9 ;
    j=9*i+1;
end 

% the overall average of all classes  per each feature 25-D
y_dd = sum( energy_responses, 1) ./ 36; 

intra_class = sum( (energy_responses - y_dd ).^2 , 1 );

inter_class = zeros(1,25);
for j =1:4
    inter_class(1,:) = inter_class(1,:) + 9 *( y_dot{j,2} - y_dd).^2 ;
end

discriminant_power = intra_class./ inter_class;

%----------------------------- PCA

[V,~,~,~,~,mu] = pca(energy_responses);
V_red = V(:,1:3);

feature_3D_train = (energy_responses-mu)*V_red;

feature_3D_test = (energy_resp_test-mu)*V_red;

%----------------------------- Plot
Fig = figure();
xyz = axes('Parent', Fig);
hold(xyz, 'all');

g1 = scatter3(feature_3D_train(1:9,1), feature_3D_train(1:9,2),feature_3D_train(1:9,3),'kd','filled');
g2 = scatter3(feature_3D_train(10:18,1), feature_3D_train(10:18,2),feature_3D_train(10:18,3),'bo','filled');
g3 = scatter3(feature_3D_train(19:27,1), feature_3D_train(19:27,2),feature_3D_train(19:27,3),'rp','filled');
g4 = scatter3(feature_3D_train(28:36,1), feature_3D_train(28:36,2),feature_3D_train(28:36,3),'gs','filled');

view(xyz, -30, 20);
grid(xyz, 'on');
xlabel('1st PC')
ylabel('2nd PC')
zlabel('3rd PC')
legend(xyz, [g1,g2,g3,g4], {'Blanket','Brick','Grass','Stones'});


%----------------------------- write to files
% 25D
csvwrite('X_train_25D.csv', energy_responses);
csvwrite('X_test_25D.csv', energy_resp_test);

% 3D
csvwrite('X_train_3D.csv', feature_3D_train);
csvwrite('X_test_3D.csv', feature_3D_test);

% test_label
for i = 1:12
    if test_label(i) == string('Blanket')
        test_label_num(i,1) = 1;
    elseif test_label(i) == string('Brick')
        test_label_num(i,1) = 2;
    elseif test_label(i) == string('Grass')
        test_label_num(i,1) = 3;
    elseif test_label(i) == string('Stones')
        test_label_num(i,1) = 4;
    end    
end
csvwrite('y_test.csv', test_label_num);


% -----------------
% class1 : Blanket
% class2 : Brick
% class3 : Grass
% class4 : Stones
% -----------------






















%-----------------------------functions---------------------------------
% read raw Images
function Imagedata = readImageRaw(filename, row, column)
    %-----------read a raw image
    f = fopen(filename);
    [rawdata, ~] = fread(f, inf);
    fclose(f);
    
    Imagedata = zeros(row, column );
    
    for i = 1 : row
        for j = 1 : column
                Imagedata(i, j) = rawdata( column * (i - 1) + (j - 1) + 1);
        end
    end
end
