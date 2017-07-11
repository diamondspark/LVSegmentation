function [slices]=know_slices(List)
lof=length(List);
for i=1:lof
patnumber=List(i,1);
vidnumber=List(i,2);
filename=sprintf('/data/Gurpreet/Echo/%d/%d(%d).dcm',patnumber,patnumber,vidnumber);
dfile=size(dicomread(filename));
slices(i,1)=dfile(1,4);
end
