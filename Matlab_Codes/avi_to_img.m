function avi_to_img(Fname)
outputFolder = '/data/Gurpreet/VC/AVIFiles';  % Change this!
% Read in the movie.
inputfname=sprintf('/data/Gurpreet/VC/Echo_avis/%d.avi',Fname);

videoObject = VideoReader(inputfname);
numberOfFrames = videoObject.NumberOfFrames;
vidHeight = videoObject.Height;
vidWidth = videoObject.Width;

numberOfFramesWritten = 0;

for frame = 1 : numberOfFrames
   % Extract the frame from the movie structure.
    thisFrame = read(videoObject, frame);
    % Create a filename.
    outputBaseFileName = sprintf('Evideo_%d_%d.jpg',Fname,frame);
    outputFullFileName = fullfile(outputFolder, outputBaseFileName);
    % Write it out to disk.
    imwrite(thisFrame, outputFullFileName, 'jpg');
end
end
