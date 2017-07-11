function view_DCM2(DCMImages,Lis)
len=length(DCMImages);
val=rem(len,4);
looplen=len-val;
for i=1:4:looplen
    figure;
    Img_1=DCMImages{i,1};
    Img_2=DCMImages{i+1,1};
    Img_3=DCMImages{i+2,1};
    Img_4=DCMImages{i+3,1};
    subplot(2,2,1); imshow(Img_1);SPTitle_1=sprintf('%s',Lis(i).name);title(SPTitle_1);
    subplot(2,2,2); imshow(Img_2);SPTitle_2=sprintf('%s',Lis(i+1).name);title(SPTitle_2);
    subplot(2,2,3); imshow(Img_3);SPTitle_3=sprintf('%s',Lis(i+2).name);title(SPTitle_3);
    subplot(2,2,4); imshow(Img_4);SPTitle_4=sprintf('%s',Lis(i+3).name);title(SPTitle_4);
end

if (val == 1)
    figure;Index=len;Img_1=DCMImages{Index,1};imshow(Img_1);SPTitle_1=sprintf('%s',Lis(Index).name);title(SPTitle_1);
end    
    
if (val == 2)
    figure;
    Index_1=len;Img_1=DCMImages{Index_1,1};subplot(1,2,1);imshow(Img_1);SPTitle_1=sprintf('%s',Lis(Index_1).name);title(SPTitle_1);
    Index_2=len-1;Img_2=DCMImages{Index_2,1};subplot(1,2,2);imshow(Img_2);SPTitle_2=sprintf('%s',Lis(Index_2).name);title(SPTitle_2);
end    
 if (val == 3)
    figure;
    Index_1=len;Img_1=DCMImages{Index_1,1};subplot(1,3,1);imshow(Img_1);SPTitle_1=sprintf('%s',Lis(Index_1).name);title(SPTitle_1);
    Index_2=len-1;Img_2=DCMImages{Index_2,1};subplot(1,3,2);imshow(Img_2);SPTitle_2=sprintf('%s',Lis(Index_2).name);title(SPTitle_2);
    Index_3=len-2;Img_3=DCMImages{Index_3,1};subplot(1,3,3);imshow(Img_3);SPTitle_3=sprintf('%s',Lis(Index_3).name);title(SPTitle_3);
 end

    
    
end
