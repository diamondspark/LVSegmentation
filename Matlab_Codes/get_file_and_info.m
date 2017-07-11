function [IMG,MANUFACTURER,WIDTH,HEIGHT]=get_file_and_info(IMG_ADDRESS,sliceno)
INFO=dicominfo(IMG_ADDRESS);
IM=dicomread(IMG_ADDRESS);
IMG=IM(:,:,:,sliceno);
MANUFACTURER=INFO.Manufacturer;
WIDTH=INFO.Width;
HEIGHT=INFO.Height;
end



