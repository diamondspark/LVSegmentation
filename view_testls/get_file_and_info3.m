
function [MANUFACTURER,WIDTH,HEIGHT]=get_file_and_info3(IMG_ADDRESS,number)
INFO=dicominfo(IMG_ADDRESS);
IMG=dicomread(IMG_ADDRESS);
savename=sprintf('/home/gus2011/tempimage_%d.mat',number);
save(savename,'IMG','-v7');
MANUFACTURER=INFO.Manufacturer;
WIDTH=INFO.Width;
HEIGHT=INFO.Height;
end



