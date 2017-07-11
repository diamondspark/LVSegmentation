function [LA]=find_size(DCM_Images)
LODIA=length(DCM_Images);
LA=int16.empty(LODIA,0);
for i=1:LODIA
    CA=DCM_Images{i,1};
    TSA=size(CA);
    
    LOTA=length(TSA);
    
    for j=1:LOTA
        LA(i,j)=TSA(1,j);
    end
end
end