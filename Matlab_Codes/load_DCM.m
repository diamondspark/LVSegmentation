function [A]=load_DCM(length,Dire,Fname)
 A=struct([]);
for i=1:length
    if (i<= 9)
        varname=sprintf('00%d',i);
       
    else
        varname=sprintf('0%d',i);
        
    end
    filename=sprintf('%s/%d(%d).dcm',Dire,Fname,i);
    img=dicomread(filename);
    A{i,1}=img;
 
end  
end