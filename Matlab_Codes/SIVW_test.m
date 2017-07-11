function SIVW_test(List)
for i=1:length(List)
Fname=List(i,1);
Img_Name=List(i,2);
%VName=List{i,3};
LDIR=sprintf('/data/Gurpreet/Echo/%d/%d(%d).dcm',Fname,Fname,Img_Name);
SDIR=sprintf('/data/Gurpreet/VC/Testing_Images');
display('============================================================');
patinfo=sprintf(' Accessing DCM files : Pat: %d Vid: %d ',Fname,Img_Name);
display(patinfo);
img=dicomread(LDIR);
Simg=size(img);
for i=1:Simg(1,4)
S_DIR=sprintf('%s/EQo_%d_%d_%d.jpg',SDIR,Fname,Img_Name,i);
imwrite(img(:,:,:,i),S_DIR)
end
end
end