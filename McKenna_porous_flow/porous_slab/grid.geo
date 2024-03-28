//

lcar0 = 0.0050;
lcar1 = 0.0125;

H = 0.10;
d = 0.02;

Point(1)  = {-15*H,   0.0, 0.0, lcar1};
Point(2)  = { -3*H,   0.0, 0.0, lcar0};
Point(3)  = { -3*H,   1*H, 0.0, lcar0};
Point(4)  = {  3*H,   1*H, 0.0, lcar0};
Point(5)  = {  3*H,   0.0, 0.0, lcar0};
Point(6)  = { 15*H,   0.0, 0.0, lcar1};
Point(7)  = { 15*H,   2*H, 0.0, lcar1};
Point(8)  = {-15*H,   2*H, 0.0, lcar1};


//Define bounding box edges
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 1};
Line(9) = {5, 2};

Transfinite Line { 9} = 71 Using Bump 0.6;

Line Loop(101) = {1:8};
Line Loop(102) = {2:4,9};

//Define unstructured far field mesh zone
Plane Surface(201) = {-101};
Plane Surface(202) = { 102};

Physical Line("inflow") = {8};
Physical Line("side") = {1,9,5,7};
Physical Line("outflow") = {6};
Physical Surface("domain") = {201,202};
