The input files are all in .csv files exported from MATLAB's SAR Model data.
    - u.csv, uc.csv, k.csv, ku0.csv, imagsRaw.csv, realsRaw.csv, imagfilteredSignal.csv
    RealfilteredSignal.csv, fastTimeFilter.csv

The sRaw data and the fastTimeFilter is imported, as well as data needed for interpolation.
This is what happens in the first lines of the main function up to the comment line
"Done Reading in values from Files"

Then the fast_time_block function is called which does everything in the Model's 
fast time block, producing fsSpotLit.

Then there is the Two-D matched filter block, which isn't made into a function yet.

From here, we can create the image with no interpolation which is saved in the 
no_interpolation.txt file.

Then the data is passed through the spatial interpolate function and is written to
the finalImage.txt file.

To run this code you need:
- All the included header files - If you are missing any, I'll email these
    + cuda_runtime
    + device_launch_parameters
    + helper_functions
    + helper_cuda
    + utils -> From Udacity course
    
- Libraries that need to be downloaded from CUDA website
    + CuFFT
    + OpenCV
        - I'm trying to remove it now, so if there are missing file issues
        they will probably come from here, if they are not from the headers above.
    + And the lib files for these

The next things should be in the project files
- The device must be set to Code Generation -> compute_20,sm_30

So basically, once the .exe is made, you need to put it in the same folder with the data files
Once you run it, there will be 2 new files finalImage, and no_interpolation.

Then you run the matlab file newrun.m, it will pull the data from finalImage.txt and no_interpolation.txt and
plot them the way the model does.
