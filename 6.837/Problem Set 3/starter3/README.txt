Submission
==========
We only provide official support for developing under the Athena cluster.
Your submission has to build and run over there to receive credit.

Please submit your entire source directory (excluding the build
directory) and compiled binary in inst/.


Question Answers
================
How do you compile and run your code? Provide instructions for Athena Linux. If your executable
requires certain parameters to work well, make sure that these are specified.

To build, run the following commands from the root directory:
$ mkdir build
$ cd build
$ cmake ..
$ make

To run the program, run the following command from the root directory:
$ ./build/a3 <integrator> <stepsize>
where <integrator> equals one of e, t, r, and <stepsize> is a number.

If <integrator> is e, then the program will use the Euler method with the given step size.
For stable results, run with <stepsize> <= 0.001.

If <integrator> is t, then the program will use the trapezoid method with the given step size.
For stable results, run with <stepsize> <= 0.01.

If <integrator> is r, then the program will use the RK4 method with the given step size.
For stable results, run with <stepsize> <= 0.01.

Pressing r while the program is running will reset the simulation to the initial state.

Extra Credit Features:
1. "Smooth" Surface
To toggle the display of the smooth surface, press s while the program is running. As you can
see, the surface has some artifacts due to the way the normals are chosen.
2. Wind
To toggle the effect of wind, press w while the program is running. The strength of the wind
changes over time.

Did you collaborate with anyone in the class? If so, let us know who you talked to and what sort of
help you gave or received.
No.

Were there any references (books, papers, websites, etc.) that you found particularly helpful for
completing your assignment? Please provide a list.
No.

Are there any known problems with your code? If so, please provide a list and, if possible, describe
what you think the cause is and how you might fix them if you had more time or motivation. This
is very important, as we’re much more likely to assign partial credit if you help us understand what’s
going on.
No.

Did you do any of the extra credit? If so, let us know how to use the additional features. If there was
a substantial amount of work involved, describe what how you did it.
Yes. Instructions for using the additional features are in the answer for the first question.

Got any comments about this assignment that you’d like to share?
No.