Synopsis
========

Pytorch reimplmentation of libMR provides core MetaRecognition and  Weibull fitting functionality.
It is used to calculate w-scores used for multi-modal fusion, renormalize SVM data and in general support open-set algorithms with outlier detection
For those looking for the C++/basic Python version look here https://github.com/Vastlab/libMR/blob/master/README.rst



Motivation
==========

Determine when something is an outlier is tricky business.
Luckily extreme value theory provides a strong basis for modeling of the largest or smallest known values and then determing if something else is too large or too small. This library supports such computations.
It is also easy to do the work in almost any advanced package (R, Matlab, etc.) but it is trick to keep it all straight.
The C++ classes here track translation and flipping to make it easier to correctly use the meta-recognition concepts.


License
=======

This version libMR is released under the BSD 3-Clause license. (see License.txt) and superseed previous license.  We dropped pursing the patent protection.

Please cite libMR in your publications if it helps your research::

  @article{Scheirer_2011_TPAMI,
    author = {Walter J. Scheirer and Anderson Rocha and Ross Michaels and Terrance E. Boult},
    title = {Meta-Recognition: The Theory and Practice of Recognition Score Analysis},
    journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI)},
    volume = {33},
    issue = {8},
    pages = {1689--1695},
    year = {2011}
  }

or one of our later works if combining it with them.

