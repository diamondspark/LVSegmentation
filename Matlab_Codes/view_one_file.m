function view_one_file(pat,num) 
fnam=sprintf('/data/Gurpreet/Echo/%d/%d(%d).dcm',pat,pat,num);
Test=dicomread(fnam);
Lenn=size(Test);
figure;
for i=1:Lenn(1,4)

    Img=Test(:,:,:,i);
pause(0.3)
imshow(Img);
end
end
