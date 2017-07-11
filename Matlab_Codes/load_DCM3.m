function [A,List]=load_DCM3(Dire,D)
 A=struct([]);
 List=struct([]);
for i=1:length(D)
filename=sprintf('%s/%s',Dire,D(i).name);
img=dicomread(filename);
A{i,1}=img;
List(i).name=D(i).name;
List(i).Identifier=i;
end  
end