function [DScore]=Cal_DiceScore(pathGT_Image,pathSEG_Image,path)

if path == 1
   GT_Image=imread(pathGT_Image);
   SEG_Image=imread(pathSEG_Image);
elseif path == 0
   GT_Image=pathGT_Image;
   SEG_Image=pathSEG_Image;
end
 
DScore=2*nnz(SEG_Image&GT_Image)./(nnz(SEG_Image)+nnz(GT_Image));
end