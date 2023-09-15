Point(1) = {-0.50, 0.00,0.0};
Point(2) = {-0.05, 0.00,0.0};
Point(3) = {-0.05,+0.05,0.0};
Point(4) = {+0.05,+0.05,0.0};
Point(5) = {+0.05, 0.00,0.0};
Point(6) = {+0.50, 0.00,0.0};
Point(7) = {+0.50,+0.50,0.0};
Point(8) = {-0.50,+0.50,0.0};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,5};
Line(5) = {5,6};
Line(6) = {6,7};
Line(7) = {7,8};
Line(8) = {8,1};

Line(10) = {5,2};


Transfinite Line {-1} = 31 Using Progression 1.10;
Transfinite Line { 5} = 31 Using Progression 1.10;

Transfinite Line { 2} = 16 Using Progression 1.0;
Transfinite Line { 4} = 16 Using Progression 1.0;

Transfinite Line { 3} = 31 Using Progression 1.0;
Transfinite Line {10} = 31 Using Progression 1.0;

Transfinite Line { 6} = 11 Using Progression 1.0;
Transfinite Line { 7} = 21 Using Progression 1.0;
Transfinite Line { 8} = 11 Using Progression 1.0;


Line Loop(11) = { 1,2,3,4,5,6,7,8};
Line Loop(12) = { 2,3,4,10};

Plane Surface(11) = {-11};
Plane Surface(12) = {12};

Physical Surface("Fluid") = {11};
Physical Surface("Sample") = {12};

Physical Curve("fluid_base") = {1,5};
Physical Curve("sample_base") = {10};
Physical Curve("outflow") = {6,7,8};
