close all;
clear;
clc;

%% read ground truth image , R100H,R100L,did,fu,gan
path='F:\jiangkui\Derain\src\test\DID_MDN_test2\BDD350\inputcrop\';
path1='F:\jiangkui\Derain\src\test\DID_MDN_test2\SEG150\NRFN3_22_1\';
path2='F:\jiangkui\Derain\src\test\DID_MDN_test2\SEG150\NRFN3_22_2\';

% mkdir(path1);
% mkdir(path2);
% list= dir(strcat(path,'*.jpg'));
% for i=1:length(list)
%     img_name=list(i).name;
%     img = imread(strcat(path,img_name));  
%     [m,n,c] = size(img);
%     img0 = img;
%     %img = modcrop(img, 8);%4£¬fu8,real8
%     %img_1 = img(1:m, 1:n/2, 1:3);%imcrop(img,[1,1,m/2,n/2]); 
%     %img_2 = img(1:m, (n/2)+1:n, 1:3);%imcrop(img,[1,1,m/2,n/2]); 
%     img_1 = imcrop(img,[1,1,639,719]);
%     img_2 = imcrop(img0,[641,1,639,719]);
%     %img_1 = single(im2double(img_1));
%     %img_2 = single(im2double(img_2));
%     imwrite(uint8(img_1),[path1,'/', img_name]);%img1_name
%     imwrite(uint8(img_2),[path2,'/', img_name]);%img1_name
% end
% 
path4='F:\jiangkui\Derain\src\test\DID_MDN_test2\SEG150\NRFN3_22\';
mkdir(path4);
list= dir(strcat(path1,'*.jpg'));
for i=1:length(list)
    img_name=list(i).name;
    img2 = imread(strcat(path1,img_name));  
    img3 = imread(strcat(path2,img_name));  
    img = [img2,img3];
    imwrite(uint8(img),[path4,'/', img_name]);%img1_name
end