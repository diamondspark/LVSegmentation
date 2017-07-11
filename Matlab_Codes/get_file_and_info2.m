function [IMG,MANUFACTURER,WIDTH,HEIGHT]=get_file_and_info(IMG_ADDRESS)
INFO=dicominfo(IMG_ADDRESS);
IMG=dicomread(IMG_ADDRESS);
MANUFACTURER=INFO.Manufacturer;
WIDTH=INFO.Width;
HEIGHT=INFO.Height;
end



