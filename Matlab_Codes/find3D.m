function [AW3DI]=find3D(Length_Array,DCM_Images)
j=1;
for i=1:length(Length_Array)
    COFA=Length_Array(i,:);
    if (COFA(1,3)~=0)&&(COFA(1,4)==0)
        AW3DI{j,1}=DCM_Images{i,1};j=j+1;
    end
end
    
       