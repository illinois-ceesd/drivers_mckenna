clM = 0.000800*2.0;
clS = 0.000600*1.4142;

Point(1) = { 0.0,  0.0, 0., 0.7*clS};

x2 = 0.0*25.4/2000;
y2 = 0.063*25.4/1000;
Point(2) = {  x2, y2, 0., clM/2.0};

x3 = 0.0;
y3 = 0.4*25.4/1000;
Point(3) = {  x3, y3, 0., clM/2.0};

x4 = 0.3*25.4/2000;
y4 = 0.4*25.4/1000;
Point(4) = {  x4, y4, 0., clM/2.0};

x5 = 0.3*25.4/2000;
y5 = 0.063*25.4/1000;
Point(5) = {  x5, y5, 0., clM/2.0};

x6 = 0.3*25.4/2000;
y6 = 0.0*25.4/1000;
Point(6) = {  x6, y6, 0., clM/2.0};


Line( 1) = {1, 2};
Line( 2) = {2, 3};
Line( 3) = {3, 4};
Line( 4) = {4, 5};
Line( 5) = {5, 6};
Line( 6) = {6, 1};


Transfinite Line { 1} =  4 Using Progression 1.0;
Transfinite Line { 2} = 18 Using Progression 1.0;
Transfinite Line { 3} =  9 Using Progression 1.0;
Transfinite Line { 4} = 18 Using Progression 1.0;
Transfinite Line { 5} =  4 Using Progression 1.0;
Transfinite Line { 6} =  9 Using Progression 1.0;

Line Loop(1) = {1,2,3,4,5,6};
Plane Surface(1) = {1};
Transfinite Surface {1} = {1,3,4,6} Alternate;


/*#########################*/

Physical Line("sym") = {1,2};
Physical Line("top") = {3};
Physical Line("right") = {4};
Physical Line("gap") = {5};
Physical Line("bottom") = {6};
Physical Surface("solid") = {1};

