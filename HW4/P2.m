clc;
clear all;
close all;

row = 512;
column = 512;

Mosaic_Img = readImageRaw('Mosaic.raw', row, column) / 255.0;
% figure(1)
% imshow(Mosaic_Img)

% % --------------------------------------------------- Part a
% 
% 
% Out_img_a = Segmentation_Process(Mosaic_Img, 63);
% 
% segmen_k_means = kmeans(Out_img_a,6);
% 
% segmentation_out = reshape(segmen_k_means, 512, 512);
%  
% output_img = Colors_Segm(segmentation_out);
% 
% blended_img = imfuse(Mosaic_Img, output_img, 'blend');
% figure(2)
% imshow(output_img)
% --------------------------------------------------- Part b

% Iblur = imgaussfilt(Mosaic_Img,1);

Mosaic_Img_partb = adapthisteq(Mosaic_Img,'clipLimit',0.1,'Distribution','rayleigh');
figure(3)
imshow(Mosaic_Img_partb)

Out_img_b = Segmentation_Process(Mosaic_Img_partb, 183).*255;

% ------------------ Energy feature averaging (train)
Out_img_b = reshape(Out_img_b,512,512,24);

size_wind = 63; % odd number
window = ones(size_wind,size_wind)./(size_wind*size_wind);

energy_responses = cell(1,24);
for i = 1:24
    pd_size = floor(size_wind/2);
    A = padarray(Out_img_b(:,:,i),[pd_size pd_size],'symmetric');

    energy_responses{1,i} = conv2(A, window,'valid') ; %mean( feature_response{i,j}.^2 , 'all');
end

X_table= zeros(512*512,24);
for i = 1:24
    X_table(:,i) = reshape(energy_responses{1,i},512*512, 1);
end

[~,score,~] = pca(X_table);

segmen_k_means = kmeans(score, 6);
segmentation_out = reshape(segmen_k_means, 512, 512);
output_img = Colors_Segm(segmentation_out);
blended_img = imfuse(Mosaic_Img, output_img, 'blend');
figure(4)
imshow(output_img)


% csvwrite('Out_img_b.csv', X_table);
% Y_segment_2 = readmatrix('Y_segmentation_2.csv');
% Y_segment_3 = readmatrix('Y_segmentation_3.csv');
% Y_segment_4 = readmatrix('Y_segmentation_4.csv');
% Y_segment_5 = readmatrix('Y_segmentation_5.csv');
% Y_segment_6 = readmatrix('Y_segmentation_6.csv');


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


function output_img = Colors_Segm(Inp_img)
R = zeros(size(Inp_img));
G = zeros(size(Inp_img));
B = zeros(size(Inp_img));

R(Inp_img==1)=107;
G(Inp_img==1)=143;
B(Inp_img==1)=159;

R(Inp_img==2)=114;
G(Inp_img==2)=99;
B(Inp_img==2)=107;

R(Inp_img==3)=175;
G(Inp_img==3)=128;
B(Inp_img==3)=74;

R(Inp_img==4)=167;
G(Inp_img==4)=57;
B(Inp_img==4)=32;

R(Inp_img==5)=144;
G(Inp_img==5)=147;
B(Inp_img==5)=104;

R(Inp_img==6)=157;
G(Inp_img==6)=189;
B(Inp_img==6)=204;

output_img = zeros([size(Inp_img),3]);
output_img(:,:,1)=R;
output_img(:,:,2)=G;
output_img(:,:,3)=B;
output_img = uint8(output_img);
end


function Out_img = Segmentation_Process(Inp_img, size_of_window)

% ----------------------------- Law's filters
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

%------------------ Filter response 
feature_response = cell(1,25);
for i= 1:25
    A = padarray(Inp_img,[2 2],'symmetric');
    feature_response{1,i} = conv2(A,filters{i,2},'valid');
end

% adjusting the r1 value (L5L5) by substracting the mean
% mean_r1 = mean( energy_responses(:,1), 1);
% fff = energy_responses(:,1) - mean_r1;


%------------------ Energy feature averaging (train)
size_wind = size_of_window; % odd number
window = ones(size_wind,size_wind)./(size_wind*size_wind);

energy_responses = cell(1,25);

for i = 1:25
    pd_size = floor(size_wind/2);
    A = padarray(feature_response{1,i},[pd_size pd_size],'symmetric');

    energy_responses{1,i} = conv2(A.^2, window,'valid') ; %mean( feature_response{i,j}.^2 , 'all');
end

X_table= zeros(512*512,25);
for i = 1:25
    X_table(:,i) = reshape(energy_responses{1,i},512*512, 1);
end

% L5L5_normaliz = mean(X_table(:,1),1);
X_table = X_table./X_table(:,1);
Out_img = X_table(:,2:end);

end