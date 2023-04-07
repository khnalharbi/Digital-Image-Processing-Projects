clc;
clear all;
close all;

row = 400;
column = 600;
BytesPerPixel = 3;

%----------------------------- read images
% RGB
Cat_1_rgb = readImageRaw('Cat_1.raw', row, column, BytesPerPixel);
Cat_2_rgb = readImageRaw('Cat_2.raw', row, column, BytesPerPixel);
Cat_Dog_rgb = readImageRaw('Cat_Dog.raw', row, column, BytesPerPixel);
Dog_1_rgb = readImageRaw('Dog_1.raw', row, column, BytesPerPixel);
Dog_2_rgb = readImageRaw('Dog_2.raw', row, column, BytesPerPixel);

% Gray
Cat_1_gray = single(rgb2gray(Cat_1_rgb));
Cat_2_gray = single(rgb2gray(Cat_2_rgb));
Cat_Dog_gray = single(rgb2gray(Cat_Dog_rgb));
Dog_1_gray = single(rgb2gray(Dog_1_rgb));
Dog_2_gray = single(rgb2gray(Dog_2_rgb));
% imshow(Cat_Dog_gray)

%% ----------------------------- Part b
%% ----------------------------- Cat 1 & Cat Dog
% Cat 1
figure(1)
imshow(Cat_1_rgb)
[feature_Cat_1, descriptor_Cat_1] = vl_sift(Cat_1_gray);
max_keyPoi_cat_1_idx = find( feature_Cat_1(3,:) == max(feature_Cat_1(3,:)) ); 
best_key_point_Cat_1 = feature_Cat_1(:,max_keyPoi_cat_1_idx);
best_descriptor_Cat_1 = descriptor_Cat_1(:,max_keyPoi_cat_1_idx);
xx = vl_plotframe( best_key_point_Cat_1 );
set(xx, 'color', 'r')

% Cat Dog
figure(2)
imshow(Cat_Dog_rgb)
[feature_Cat_Dog, descriptor_Cat_Dog] = vl_sift(Cat_Dog_gray);
Knn_index = knnsearch( feature_Cat_Dog', best_key_point_Cat_1');
xx = vl_plotframe( feature_Cat_Dog(:,Knn_index) );
set(xx, 'color', 'r')

% detectSiftFeatures
% [MATCHES,SCORES] = vl_ubcmatch(descriptor_Cat_1, descriptor_Cat_Dog);


%% ----------------------------- Dog 1 & Cat Dog
% Dog 1
figure(3)
imshow(Dog_1_rgb)
[feature_Dog_1, descriptor_Dog_1] = vl_sift(Dog_1_gray);
max_keyPoi_Dog_1_idx = find( feature_Dog_1(3,:) == max(feature_Dog_1(3,:)) ); 
best_key_point_Dog_1 = feature_Dog_1(:,max_keyPoi_Dog_1_idx);
best_descriptor_Dog_1 = descriptor_Dog_1(:,max_keyPoi_Dog_1_idx);
xx = vl_plotframe( best_key_point_Dog_1 );
set(xx, 'color', 'r')


% Cat Dog
figure(4)
imshow(Cat_Dog_rgb)
Knn_index = knnsearch( feature_Cat_Dog', best_key_point_Dog_1');
best_key_point_Cat_Dog = feature_Cat_Dog(:,Knn_index);
xx = vl_plotframe( feature_Cat_Dog(:,Knn_index) );
set(xx, 'color', 'r')


%%  ----------------------------- Cat 1 & Cat 2
% Cat 1
figure(5)
imshow(Cat_1_rgb)
xx = vl_plotframe( best_key_point_Cat_1 );
set(xx, 'color', 'r')

% Cat 2
figure(6)
imshow(Cat_2_rgb)
[feature_Cat_2, descriptor_Cat_2] = vl_sift(Cat_2_gray);
Knn_index = knnsearch( feature_Cat_2', best_key_point_Cat_1');
best_key_point_Cat_2 = feature_Cat_2(:,Knn_index);
xx = vl_plotframe( feature_Cat_2(:,Knn_index) );
set(xx, 'color', 'r')

%% ----------------------------- Cat 1 & Dog 1

% Cat 1
figure(7)
imshow(Cat_1_rgb)
xx = vl_plotframe( best_key_point_Cat_1 );
set(xx, 'color', 'r')

% Dog 1
figure(8)
imshow(Dog_1_rgb)
[feature_Dog_1, descriptor_Dog_1] = vl_sift(Dog_1_gray);
Knn_index = knnsearch( feature_Dog_1', best_key_point_Cat_1');
xx = vl_plotframe( feature_Dog_1(:,Knn_index) );
set(xx, 'color', 'r')
%%
% Dog 2
figure(8)
imshow(Dog_2_rgb)
[feature_Dog_2, descriptor_Dog_2] = vl_sift(Dog_2_gray);

%% -------------------- Part c


% rng(00965);  
descriptor_images = double([descriptor_Cat_1';descriptor_Cat_2';descriptor_Dog_1';descriptor_Cat_Dog']);

lenght_1 = size(descriptor_Cat_1,2);
lenght_2 = size(descriptor_Cat_2,2);
lenght_3 = size(descriptor_Dog_1,2);
lenght_4 = size(descriptor_Cat_Dog,2);

[codebook_dic, centroid] = kmeans(descriptor_images,8);





Cat_1_BoW = codebook_dic(1:lenght_1);
lenght_1 = lenght_1+1;

Cat_2_BoW = codebook_dic(lenght_1:(lenght_1+lenght_2-1));
lenght_2 = lenght_2+lenght_1;

Dog_1_BoW = codebook_dic(lenght_2:(lenght_2+lenght_3-1));
lenght_3 = lenght_3+lenght_2;

cat_dog_idx = codebook_dic(lenght_3:(lenght_3+lenght_4-1));

Dog_2_BoW = knnsearch(centroid, descriptor_Dog_2');


sumcat = sum(hist(Cat_1_BoW, 8));
sumdog1 = sum(hist(Dog_1_BoW, 8));
sumdog2 = sum(hist(Dog_2_BoW, 8));
ALL = [ hist(Cat_1_BoW, 8)./sumcat; hist(Dog_1_BoW, 8)./sumdog1; hist(Dog_2_BoW, 8)./sumdog2] ; 

minimal_cat_1_Dog_2 = min(ALL([1,3],:), [], 1);
maximal_cat_1_Dog_2 = max(ALL([1,3],:), [], 1);
sim_cat1_dog2 = sum(minimal_cat_1_Dog_2) / sum(maximal_cat_1_Dog_2)


minimal_Dog_1_Dog_2 = min(ALL([2,3],:), [], 1);
maximal_Dog_1_Dog_2 = max(ALL([2,3],:), [], 1);
sim_dog1_dog2 = sum(minimal_Dog_1_Dog_2) / sum(maximal_Dog_1_Dog_2)




%% --------------------------functions---------------------------------
% read raw Images
function Imagedata = readImageRaw(filename, row, column, BytesPerPixel)
    %-----------read a raw image
    f = fopen(filename);
    [rawdata, ~] = fread(f, inf);
    fclose(f);
    
    Imagedata = zeros(row, column, BytesPerPixel);
    
    for i = 1 : row
        for j = 1 : column
            for k = 1 : BytesPerPixel
                Imagedata(i, j, k) = rawdata(3 * (column * (i - 1) + (j - 1)) + k) / 255.0;
            end
        end
    end
end

