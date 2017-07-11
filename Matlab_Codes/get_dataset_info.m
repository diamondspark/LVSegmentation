function [Fileinfo]=get_dataset_info(FolderList)
filenum=0;
for i=1:length(FolderList)
Fname=FolderList(i,1);
DIR=sprintf('/data/Gurpreet/Echo/%d',Fname);
display('====================');
display(DIR);
D = dir([DIR, '/*.dcm']);
Num = length(D(not([D.isdir])));
%display('Checking directory for dcm files');
for j=1:Num
    filenum=filenum+1;
    readfilename=sprintf('%s/%s',DIR,D(j).name);
    dicomreadfile=dicominfo(readfilename);
    dicomfile=dicomread(readfilename);
    namestring=dicomreadfile.Filename;
    mid1=strsplit(namestring,'/');
    mid2=strsplit(mid1{1,6},'(');
    Patientname=mid2{1,1};
    mid3=strsplit(mid2{1,2},')');
    Filename=mid3{1,1};
    Fileinfo(filenum).Patientname=Patientname;
    Fileinfo(filenum).Filename=Filename;
    Fileinfo(filenum).NoF=size(dicomfile,4);
    Fileinfo(filenum).Manufacturer=dicomreadfile.Manufacturer;
    Fileinfo(filenum).Width=dicomreadfile.Width;
    Fileinfo(filenum).Height=dicomreadfile.Height;
end
end
end
