function save_images_echo_4_D(Fname)

DIR=sprintf('/data/Gurpreet/Echo/%d',Fname);
SDIR=sprintf('/data/Gurpreet/Echo/%d/Images/',Fname);

display('====================');
D = dir([DIR, '/*.dcm']);
Num = length(D(not([D.isdir])));
display('Checking directory for dcm files');
[A]=load_DCM(Num,DIR,Fname);
[LA]=find_size(A);
[AW4DI]=find4D(LA,A);
[LA4DI]=find_size(AW4DI);
display('Making new directory');
cmnd=sprintf('sudo mkdir %s',SDIR);
system(cmnd);
display('Saving Images');
for i=1:length(LA4DI)
for j=1:LA4DI(i,4)
name=sprintf('%sTEE_%d_%d_%d.jpg',SDIR,Fname,i,j);
imwrite(AW4DI{i,1}(:,:,j),name)
end
end


end