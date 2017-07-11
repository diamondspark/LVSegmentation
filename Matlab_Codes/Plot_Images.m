function [AW4DI,LIST]=Plot_Images(Fname)
DIR=sprintf('/data/Gurpreet/Echo/%d',Fname);

display('====================');
D = dir([DIR, '/*.dcm']);
Num = length(D(not([D.isdir])));
display('Checking directory for dcm files');
[A]=load_DCM3(DIR,D);
[LA]=find_size(A);
[AW4DI,LIST]=find4D(LA,A,D);
for i=1:length(AW4DI)
DCMImages{i,1}=AW4DI{i,1}(:,:,:,5);
end
view_DCM2(DCMImages,LIST);