clF = 0.001600*2.0000;
clB = 0.000800*2.0;
clM = 0.000800*2.0;
clS = 0.000600*1.4142;
cl0 = 0.007200*2.0;
cl1 = 0.024000*2.0;
cl2 = 0.072000*2.0;

radius = 1.0;

int_radius = 2.38*25.4/2000;
ext_radius = 2.89*25.4/2000;
burner_base_radius = 4.725*25.4/2000;

burner_height = 0.10;
flame_dist = 0.0025;

mat_location = burner_height + 0.010;  //arbitrary value
mat_wide = 0.5*1.25*25.4/1000;
mat_thick = 15/32*25.4/1000;
insulator_1 = 2.0*25.4/1000;
insulator_2 = 3.0*25.4/1000;
mat_holder = 4.0*25.4/1000;
mat_BL_layer_x = 0.0030;
mat_BL_layer_y = 0.0030;


Point(2) = {ext_radius, burner_height, 0.0, 0.000200};
Point(3) = {int_radius, burner_height, 0.0, 1.0};
Point(4) = {mat_wide + mat_BL_layer_x, burner_height, 0.0, 1.0};
Point(5) = {               0., burner_height, 0.0, 1.0};

delta = (mat_location - mat_BL_layer_y - burner_height - flame_dist);
flame_pos = burner_height + flame_dist;

Point(9) = {     0.0, mat_location - mat_BL_layer_y, 0.,     clS};
Point(10) = {     0.0, mat_location                 , 0., 0.7*clS};
x3 = 0.3*25.4/2000;
Point(11) = {      x3, mat_location                 , 0., 0.7*clS};
Point(12) = {mat_wide, mat_location                 , 0.,     1.0};
Point(13) = {mat_wide, mat_location + mat_thick     , 0.,     1.0};
Point(14) = {mat_wide, mat_location + insulator_1   , 0.,     1.0};

Point(15) = {mat_wide,                  mat_location + 1.*mat_holder, 0., 1.};
Point(16) = {     0.0,                  mat_location + 1.*mat_holder, 0., clM};
Point(17) = {     0.0, mat_BL_layer_y + mat_location + 1.*mat_holder, 0., 2.0*clS};
Point(18) = {     0.0,                                          0.45, 0., 7.5*clS};

Point(20) = {              0.00, radius, 0., 0.20*cl2};
Point(21) = {              0.25, radius, 0., 0.30*cl2};
Point(22) = {            radius, radius, 0.,      cl2};
Point(24) = {            radius,     0., 0.,      cl2};
Point(25) = {burner_base_radius,     0., 0., 6.66*clB};

aux = 0.5*(burner_base_radius + ext_radius);
Point(26) = {burner_base_radius, burner_height-0.0110, 0.0, clB};
Point(27) = {               aux, burner_height-0.0055, 0.0, clB};

offset = 0.0;
Point(28) = {       burner_base_radius, burner_height + flame_dist, 0., 1.0};
Point(29) = {                      aux, burner_height + flame_dist, 0., 1.0};
Point(30) = {      ext_radius + offset, burner_height + flame_dist, 0., 1.0};
Point(31) = {               int_radius, burner_height + flame_dist, 0., 1.0};
Point(32) = {mat_wide + mat_BL_layer_x, burner_height + flame_dist, 0., 1.0};
Point(33) = {                       0., burner_height + flame_dist, 0., clS};


Point(41) = {mat_wide + mat_BL_layer_x,            - mat_BL_layer_y + mat_location, 0., 1.};
Point(43) = {mat_wide + mat_BL_layer_x,                   mat_thick + mat_location, 0., 1.};
Point(44) = {mat_wide + mat_BL_layer_x,                 insulator_1 + mat_location, 0., 1.};
Point(45) = {mat_wide + mat_BL_layer_x, mat_holder + mat_BL_layer_y + mat_location, 0., 1.};


x1 = 0.0;
y1 = 0.4*25.4/1000;
Point(51) = {  x1, mat_location + y1, 0., clM/2.0};

x2 = 0.3*25.4/2000;
y2 = 0.4*25.4/1000;
Point(52) = {  x2, mat_location + y2, 0., clM/2.0};

x3 = 0.3*25.4/2000;
y3 = 0.063*25.4/1000;
Point(53) = {  x3, mat_location + y3, 0., clM/2.0};

x4 = 0.35*25.4/2000;
y4 = 0.063*25.4/1000;
Point(54) = {  x4, mat_location + y4, 0., clM/2.0};

x5 = 0.35*25.4/2000;
Point(55) = {  x1, mat_location + insulator_2, 0., 2*clM/2.0};

/*x6 = 0.35*25.4/2000;*/
/*Point(56) = {  x6, mat_location + mat_holder + mat_BL_layer_y, 0., clM/2.0};*/

x7 = 0.35*25.4/2000;
Point(57) = {  x7, mat_location + insulator_1, 0., 2*clM/2.0};

x8 = 0.5*25.4/2000;
Point(58) = {  x8, mat_location + insulator_1, 0., 2*clM/2.0};

x9 = 0.5*25.4/2000;
Point(59) = {  x9, mat_location + insulator_2, 0., 2*clM/2.0};


Line( 1) = {2,  3};
Line( 2) = {3,  4};
Line( 3) = {3, 31};
Line( 4) = {4, 32};
Line( 5) = {4,  5};
Line( 6) = {5, 33};

Line(10) = {26, 28};
Line(11) = {28, 29};
Line(12) = {29, 30};
Line(13) = {27, 29};
Line(14) = {26, 27};
Line(15) = {27,  2};
Line(16) = { 2, 30};
Line(17) = {30, 31};
Line(18) = {31, 32};
Line(19) = {32, 33};

Line(22) = {33,  9};
Line(23) = { 9, 41};
Line(26) = {41, 43};
Line(27) = {43, 44};
Line(28) = {44, 45};
Line(29) = {45, 17};
Line(30) = {17, 18};
Line(31) = {18, 20};

Line(40) = {20, 21};
Line(41) = {21, 22};
Line(42) = {22, 24};
Line(43) = {24, 25};
Line(44) = {25, 26};

Line(49) = { 9, 10};
Line(50) = {10, 11};
Line(51) = {11, 12};
Line(52) = {12, 41};
Line(53) = {13, 43};
Line(54) = {12, 13};
Line(55) = {13, 14};
Line(56) = {14, 15};
Line(57) = {15, 16};
Line(58) = {45, 15};
Line(59) = {16, 17};
Line(60) = {44, 14};

Line(61) = {10, 51};
Line(62) = {51, 52};
Line(63) = {52, 53};
Line(64) = {53, 54};
Line(65) = {54, 57};
/*Line(66) = {55, 56};*/
Line(67) = {55, 16};
Line(68) = {55, 51};
Line(69) = {57, 58};
Line(70) = {58, 59};
Line(71) = {59, 55};
Line(75) = {53, 11};
Line(76) = {11, 53};

/*horizontal*/
Transfinite Line {  1} = 10 Using Bump 0.75;
Transfinite Line { 17} = 10 Using Bump 0.75;
Transfinite Line {  2} = 18 Using Progression 1.02;
Transfinite Line { 18} = 18 Using Progression 1.02;

Transfinite Line {- 5} = 26 Using Progression 1.00;
Transfinite Line {-19} = 26 Using Progression 1.00;
Transfinite Line { 23} = 28 Using Progression 1.00;
Transfinite Line {-50} =  7 Using Progression 1.00;
Transfinite Line {-51} = 22 Using Progression 1.01;

Transfinite Line { 26} = 19 Using Progression 1.024;
Transfinite Line { 54} = 19 Using Progression 1.055;

/*vertical*/
Transfinite Line {  3} = 14 Using Progression 1.1;
Transfinite Line { 16} = 14 Using Progression 1.1;
Transfinite Line {  4} = 14 Using Progression 1.1;
Transfinite Line {  6} = 14 Using Progression 1.1;
Transfinite Line { 13} = 14 Using Progression 1.0;
Transfinite Line { 10} = 14 Using Progression 1.0;

/*burner*/
Transfinite Line {-15} = 15 Using Progression 1.04;
Transfinite Line {-12} = 15 Using Progression 1.04;
Transfinite Line { 14} = 11 Using Progression 1.00;
Transfinite Line { 11} = 11 Using Progression 1.00;

/*material*/
Transfinite Line {-49} = 10 Using Progression 1.100;
Transfinite Line { 52} = 10 Using Progression 1.100;
Transfinite Line { 53} = 10 Using Progression 1.100;
Transfinite Line {-58} = 10 Using Progression 1.100;
Transfinite Line {-60} = 10 Using Progression 1.100;
Transfinite Line { 66} = 10 Using Progression 1.100;
Transfinite Line { 59} = 10 Using Progression 1.100;

/*Transfinite Line { 67} = 3 Using Progression 1.020;*/
/*Transfinite Line { 30} = 3 Using Progression 1.020;*/
/*Transfinite Line { 66} = 31 Using Progression 1.033;*/

Transfinite Line { 27} = 37 Using Progression 1.0;
Transfinite Line { 55} = 37 Using Progression 1.0;

Transfinite Line { 28} = 41 Using Progression 1.008;
Transfinite Line { 56} = 41 Using Progression 1.005;

Transfinite Line { 57} = 11 Using Progression 1.00;
Transfinite Line {-29} = 11 Using Progression 1.027;

Transfinite Line { 69} = 2;
Transfinite Line { 75} = 4;
Transfinite Line { 76} = 4;

//+
Field[1] = Box;
Field[1].XMax = 0.08;
Field[1].XMin = 0.0;
Field[1].YMax = 0.25;
Field[1].YMin = 0.105;
Field[1].ZMax = -1;
Field[1].ZMin = 1;
Field[1].Thickness = 0.8;
Field[1].VIn = 0.002;
Field[1].VOut = 0.12;

Field[2] = Box;
Field[2].XMax = 0.12;
Field[2].XMin = 0.0;
Field[2].YMax = 0.375;
Field[2].YMin = 0.1;
Field[2].ZMax = -1;
Field[2].ZMin = 1;
Field[2].Thickness = 0.8;
Field[2].VIn = 0.004;
Field[2].VOut = 0.12;

Field[3] = Box;
Field[3].XMax = 0.16;
Field[3].XMin = 0.0;
Field[3].YMax = 0.5;
Field[3].YMin = 0.2;
Field[3].ZMax = -1;
Field[3].ZMin = 1;
Field[3].Thickness = 0.8;
Field[3].VIn = 0.008;
Field[3].VOut = 0.12;

Field[4] = Box;
Field[4].XMax = 0.05;
Field[4].XMin = 0.016;
Field[4].YMax = 0.16;
Field[4].YMin = 0.10;
Field[4].ZMax = -1;
Field[4].ZMin = 1;
Field[4].Thickness = 0.01;
Field[4].VIn = 0.001;
Field[4].VOut = 0.12;

Field[5] = Box;
Field[5].XMax = 0.065;
Field[5].XMin = 0.016;
Field[5].YMax = 0.215;
Field[5].YMin = 0.10;
Field[5].ZMax = -1;
Field[5].ZMin = 1;
Field[5].Thickness = 0.01;
Field[5].VIn = 0.0015;
Field[5].VOut = 0.12;


Field[6] = Min;
Field[6].FieldsList = {1, 2, 3, 4, 5};
Background Field = 6;


Line Loop(40) = {10:12,17:19,22:23,26:31,40:44};
Plane Surface(1) = {40};

Line Loop(41) = {62:65,69:71,68};
Plane Surface(2) = {-41};

Point(101) = {             0.000500, flame_dist + burner_height, 0., 1.0};
Point(102) = {int_radius + 0.000281,              burner_height, 0., 1.0};
Point(103) = {int_radius - 0.000281,              burner_height, 0., 1.0};
Point(104) = {int_radius           ,   0.000100 + burner_height, 0., 1.0};
/*Point(105) = {      0.005 - 0.00050, flame_dist + burner_height, 0., 1.0};*/
/*Point(106) = {      0.005 + 0.00050, flame_dist + burner_height, 0., 1.0};*/
Point(107) = {        0.           ,  -0.000200 + mat_location , 0., 1.0};

Line Loop(42) = {1,3,-17,-16};
Plane Surface(3) = {42};
Transfinite Surface {3} Alternate;

Line Loop(43) = {4,-18,-3,2};
Plane Surface(4) = {43};
Transfinite Surface {4} Alternate;

Line Loop(44) = {-4,5,6,-19};
Plane Surface(5) = {44};
Transfinite Surface {5} Alternate;

Line Loop(45) = {15,16,-12,-13};
Plane Surface(6) = {45};
Transfinite Surface {6} Alternate;

Line Loop(46) = {14,13,-11,-10};
Plane Surface(7) = {46};
Transfinite Surface {7} Alternate;


Line Loop(50) = {49,50,51,52,-23};
Plane Surface(11) = {50};
Transfinite Surface {11} = {10,12,41,9};

Line Loop(51) = {-53, -54, 52, 26};
Plane Surface(12) = {-51};
Transfinite Surface {12};

Line Loop(52) = {-60,-27, -53, 55};
Plane Surface(13) = {52};
Transfinite Surface {13};

Line Loop(53) = {60, 56, -58, -28};
Plane Surface(14) = {53};
Transfinite Surface {14} Right;

Line Loop(54) = {59,-29, 58, 57};
Plane Surface(15) = {54};
Transfinite Surface {15};

Line Loop(60) = {61:63,75,-50};
Plane Surface(20) = {60};

Line Loop(61) = {64,65,69:71,67,-57,-56,-55,-54,-51,76};
Plane Surface(21) = {61};

/*#########################*/

Physical Line("inlet") = {1,2,5};
Physical Line("symmetry") = {6,22,49,30,31,59,68};
Physical Line("burner") = {44,14,15};
Physical Line("linear") = {41,42,43};
Physical Line("outflow") = {40};
Physical Line("wall_sym") = {61,67};
Physical Line("wall_gap") = {75,76};
Physical Surface("fluid") = {1:7,11:15};
Physical Surface("solid") = {20,21};
