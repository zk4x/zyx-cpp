This is a tensor library

It is written in c++, with accelerators using possibly other languages (such as opencl)
It is header only library

Requirements
This library relies on STL, thus you will need to have STL installed
By default it runs on cpu
In order to use other accelerators, you will need to meet their respective requirements,
see accelerators.txt

In order to use the library, just include src directory when compiling
(-I/path-to-this-library/src)

For examples, see folder examples
In order to run examples, go to examples directory, run make, that will compile all the examples,
and then you can run them by launching their executable files that will appear in examples
directory

For tests, see folder tests
In order to run tests, follow the same procedure as with examples

To see how this library works, read how_it_works.txt
