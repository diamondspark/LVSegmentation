function view_4D(MAT,time)
SA=size(MAT);
LN=SA(1,4);
for i=1:LN
imshow(MAT(:,:,:,i));pause(time);
end
end