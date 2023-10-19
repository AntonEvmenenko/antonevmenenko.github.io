setlocal

set input_file=input.mp4
set output_file=output.gif

set crop=4 & :: pixels
set border_thickness=2 & :: pixels
set skip=1 & :: seconds

set /a temp=%border_thickness% * 2

ffmpeg -i %input_file% -ss %skip% -filter_complex "[0]crop=w=iw-%crop%:h=ih-%crop%[a];[a]pad=w=%temp%+iw:h=%temp%+ih:x=%border_thickness%:y=%border_thickness%:color=gray" %output_file%

endlocal