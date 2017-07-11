function [AW4DI,List]=find4D(Length_Array,DCM_Images,DD)
j=1;
List=struct([]);
for i=1:length(Length_Array)
    COFA=Length_Array(i,:);
    if (COFA(1,3)~=0)&&(COFA(1,4)~=0)
        AW4DI{j,1}=DCM_Images{i,1};
        List(j).name=DD(i).name;
        List(j).Identifier=j;j=j+1;
    end
end
    
        