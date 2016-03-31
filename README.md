eogert
======

EOGERT - EOG Event Recognizer Tool

Eogert is a probabilistic online method for detecting fixations, saccades, and blinks from EOG signal.
The beginning of the signal (the first 'TRAIN_SECS' seconds) is dedicated to training. This period should contain at least few random saccades and blinks.

The implementation is written in Matlab and it uses normpdf routine, found in the statistics toolbox; if your don't have it, don't bother to buy it but just make your own with 

norm_pdf = exp(-0.5 * ((x - mu)./sigma).^2) ./ (sqrt(2 * pi) * sigma).

The main file is *eogert.m* which calls *EMgauss1D.m*. In the current code, the EOG signals are loaded from the *TEST001.mat* file which must therefore also exist in the same folder 
and the signals are read "realtime" from the loaded files. Modify the corresponding code to read the signals over some actual stream (such as socket).

There is also an offline version available, named *eogert_offline*. You must input EOG signals to it. The output is saved into a file. Type "help eogert_offline" for further details.

The method is described in<br>
Toivanen, M., Pettersson, K., Lukander, K. (2015). A probabilistic real-time algorithm for detecting blinks, saccades, and fixations from EOG data. Journal of Eye Movement Research, 8(2):1,1-14.
