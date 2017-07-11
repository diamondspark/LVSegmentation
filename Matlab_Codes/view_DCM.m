 function view_DCM(DCMImages,Rows,Columns)
len=length(DCMImages);
TF=Rows*Columns;
spn=1;
for i=1:4:len
    figure;
    Img_1=DCMImages{i,1};
    Img_2=DCMImages{i+1,1};
    Img_3=DCMImages{i+2,1};
    Img_4=DCMImages{i+3,1};
        subplot(Columns,Rows,spn); imshow(Img);
     
 end     
