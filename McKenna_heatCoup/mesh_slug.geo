clM = 0.000800*2.0;
clS = 0.000600*1.4142;

Point(1) = { 0.0,  0.0, 0., 0.7*clS};

x1 = 0.0;
y1 = 0.4*25.4/1000;
Point(2) = {  x1, y1, 0., clM/2.0};

x2 = 0.3*25.4/2000;
y2 = 0.4*25.4/1000;
Point(3) = {  x2, y2, 0., clM/2.0};

x3 = 0.3*25.4/2000;
y3 = 0.0*25.4/1000;
Point(4) = {  x3, y3, 0., clM/2.0};


Line( 1) = {1, 2};
Line( 2) = {2, 3};
Line( 3) = {3, 4};
Line( 4) = {4, 1};


Transfinite Line {1} = 21 Using Progression 1.0;
Transfinite Line {2} = 6 Using Progression 1.0;
Transfinite Line {3} = 21 Using Progression 1.0;
Transfinite Line {4} = 6 Using Progression 1.0;

Line Loop(1) = {1,2,3,4};
Plane Surface(1) = {1};
Transfinite Surface {1} Alternate;


/*#########################*/

Physical Line("sym") = {1};
Physical Line("top") = {2};
Physical Line("right") = {3};
Physical Line("bottom") = {4};
Physical Surface("solid") = {1};

