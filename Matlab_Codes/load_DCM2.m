function [A]=load_DCM2(List,Dire,DforDCM)
A=struct([]);
for i=1:length(List)
filename=sprintf('%s/%s',Dire,DforDCM(List(i,1)).name);
%output=sprintf('Loaded %s',DforDCM(List(i,1)).name);
%display(output);
img=dicomread(filename);
A{i,1}=img;
end  
end