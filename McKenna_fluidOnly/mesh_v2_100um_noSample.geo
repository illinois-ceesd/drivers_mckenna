clF = 0.001600*2.0000;
clB = 0.000800*2.0;
clM = 0.000800*2.0;
clS = 0.000600*1.4142;
cl0 = 0.007200*2.0;
cl1 = 0.024000*2.0;
cl2 = 0.072000*2.0;

radius = 1.5;

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
/*Point(10) = {     0.0, mat_location                 , 0., 0.7*clS};*/
x3 = 0.3*25.4/2000;

/*Point(16) = {     0.0,                  mat_location + 1.*mat_holder, 0., clM};*/
Point(17) = {     0.0, mat_BL_layer_y + mat_location + 1.*mat_holder, 0., 2.0*clS};
Point(18) = {     0.0,                                          0.55, 0., 7.5*clS};

Point(20) = {              0.00, radius, 0., 0.20*cl2};
Point(21) = {              0.25, radius, 0., 0.20*cl2};
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


/*Point(41) = {mat_wide + mat_BL_layer_x,            - mat_BL_layer_y + mat_location, 0., 1.};*/
/*Point(43) = {mat_wide + mat_BL_layer_x,                   mat_thick + mat_location, 0., 1.};*/
/*Point(44) = {mat_wide + mat_BL_layer_x,                 insulator_1 + mat_location, 0., 1.};*/
/*Point(45) = {mat_wide + mat_BL_layer_x, mat_holder + mat_BL_layer_y + mat_location, 0., 1.};*/



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
/*Line(23) = { 9, 41};*/
/*Line(26) = {41, 43};*/
/*Line(27) = {43, 44};*/
/*Line(28) = {44, 45};*/
/*Line(29) = {45, 17};*/
Line(30) = {17, 18};
Line(31) = {18, 20};

Line(40) = {20, 21};
Line(41) = {21, 22};
Line(42) = {22, 24};
Line(43) = {24, 25};
Line(44) = {25, 26};

Line(49) = { 9, 17};
/*Line(50) = {10, 11};*/
/*Line(51) = {11, 12};*/
/*Line(52) = {12, 41};*/
/*Line(53) = {13, 43};*/
/*Line(54) = {12, 13};*/
/*Line(55) = {13, 14};*/
/*Line(56) = {14, 15};*/
/*Line(57) = {15, 16};*/
/*Line(58) = {45, 15};*/
/*Line(59) = {16, 17};*/
/*Line(60) = {44, 14};*/

/*Line(61) = {10, 51};*/
/*Line(62) = {51, 52};*/
/*Line(63) = {52, 53};*/
/*Line(64) = {53, 54};*/
/*Line(65) = {54, 57};*/
/*Line(67) = {55, 16};*/
/*Line(68) = {55, 51};*/
/*Line(69) = {57, 58};*/
/*Line(70) = {58, 59};*/
/*Line(71) = {59, 55};*/
/*Line(75) = {53, 11};*/
/*Line(76) = {11, 53};*/

/*horizontal*/
Transfinite Line {  1} = 10 Using Bump 0.75;
Transfinite Line { 17} = 10 Using Bump 0.75;
Transfinite Line {  2} = 18 Using Progression 1.02;
Transfinite Line { 18} = 18 Using Progression 1.02;


Transfinite Line { 22} = 7 Using Progression 1.00;

Transfinite Line {- 5} = 26 Using Progression 1.00;
Transfinite Line {-19} = 26 Using Progression 1.00;
Transfinite Line {-50} =  7 Using Progression 1.00;
Transfinite Line {-51} = 22 Using Progression 1.01;

Transfinite Line { 26} = 17 Using Progression 1.020;
Transfinite Line { 54} = 17 Using Progression 1.050;

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
/*Transfinite Line {-49} = 10 Using Progression 1.100;*/
/*Transfinite Line { 52} = 10 Using Progression 1.100;*/
/*Transfinite Line { 53} = 10 Using Progression 1.100;*/
/*Transfinite Line {-58} = 10 Using Progression 1.100;*/
/*Transfinite Line {-60} = 10 Using Progression 1.100;*/
/*Transfinite Line { 66} = 10 Using Progression 1.100;*/
/*Transfinite Line { 59} = 10 Using Progression 1.100;*/

/*Transfinite Line { 67} = 3 Using Progression 1.020;*/
/*Transfinite Line { 30} = 3 Using Progression 1.020;*/
/*Transfinite Line { 66} = 31 Using Progression 1.033;*/

Transfinite Line { 27} = 27 Using Progression 1.02;
Transfinite Line { 55} = 27 Using Progression 1.02;

Transfinite Line {-28} = 25 Using Progression 1.005;
Transfinite Line {-56} = 25 Using Progression 1.014;

Transfinite Line { 57} = 11 Using Progression 1.00;
Transfinite Line {-29} = 11 Using Progression 1.027;

Transfinite Line { 69} = 2;
Transfinite Line { 75} = 4;
Transfinite Line { 76} = 4;

Line Loop(40) = {10:12,17:19,22,49,30:31,40:44};
Plane Surface(1) = {40};

/*Line Loop(41) = {62:65,69:71,68};*/
/*Plane Surface(2) = {-41};*/

Point(70) = {  0.080, burner_height - 0.028, 0., 2.250*clF};
Point(71) = {  0.120, burner_height - 0.025, 0., 2.500*clF};
Point(72) = {  0.160, burner_height - 0.020, 0., 2.625*clF};
Point(73) = {  0.200, burner_height - 0.012, 0., 2.750*clF};
Point(74) = {  0.230, burner_height + 0.000, 0., 2.875*clF};
Point(75) = {  0.260, burner_height + 0.020, 0., 3.000*clF};
Point(76) = {  0.280, burner_height + 0.050, 0., 3.125*clF};
Point(77) = {  0.300, burner_height + 0.080, 0., 3.250*clF};
Point(78) = {  0.320, burner_height + 0.120, 0., 3.375*clF};
Point(79) = {  0.330, burner_height + 0.160, 0., 3.500*clF};
Point(80) = {  0.340, burner_height + 0.200, 0., 3.625*clF};
Point(81) = {  0.345, burner_height + 0.240, 0., 3.750*clF};
Point(82) = {  0.350, burner_height + 0.280, 0., 4.000*clF};
Point(83) = {  0.350, burner_height + 0.320, 0., 4.250*clF};
Point(84) = {  0.350, burner_height + 0.360, 0., 4.500*clF};
Point(85) = {  0.350, burner_height + 0.400, 0., 4.750*clF};
Point(86) = {  0.350, burner_height + 0.500, 0., 5.000*clF};
Point(87) = {  0.350, burner_height + 0.600, 0., 5.500*clF};
Point(88) = {  0.350, burner_height + 0.700, 0., 6.000*clF};
Point(89) = {  0.350, burner_height + 0.800, 0., 6.000*clF};
/*Point(90) = {  0.250, burner_height + 0.900, 0., 6.000*clF};*/
Point{70:89} In Surface{1};

Point(101) = {             0.000500, flame_dist + burner_height, 0., 1.0};
Point(102) = {int_radius + 0.000281,              burner_height, 0., 1.0};
Point(103) = {int_radius - 0.000281,              burner_height, 0., 1.0};
Point(104) = {int_radius           ,   0.000100 + burner_height, 0., 1.0};
/*Point(107) = {        0.           ,  -0.000200 + mat_location , 0., 1.0};*/

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


/*Line Loop(50) = {49,50,51,52,-23};*/
/*Plane Surface(11) = {50};*/
/*Transfinite Surface {11} = {10,12,41,9};*/

/*Line Loop(51) = {-53, -54, 52, 26};*/
/*Plane Surface(12) = {-51};*/
/*Transfinite Surface {12};*/

/*Line Loop(52) = {-60,-27, -53, 55};*/
/*Plane Surface(13) = {52};*/
/*Transfinite Surface {13};*/

/*Line Loop(53) = {60, 56, -58, -28};*/
/*Plane Surface(14) = {53};*/
/*Transfinite Surface {14} Right;*/

/*Line Loop(54) = {59,-29, 58, 57};*/
/*Plane Surface(15) = {54};*/
/*Transfinite Surface {15};*/

/*Line Loop(60) = {61:63,75,-50};*/
/*Plane Surface(20) = {60};*/

/*Line Loop(61) = {64,65,69:71,67,-57,-56,-55,-54,-51,76};*/
/*Plane Surface(21) = {61};*/

/*#########################*/

Physical Line("inlet") = {1,2,5};
Physical Line("outlet") = {40};
Physical Line("symmetry") = {6,22,49,30,31,59,68};
Physical Line("burner") = {44,14,15};
Physical Line("linear") = {41,42,43};
Physical Surface("fluid") = {1:7};

Mesh 2;
order = 1;
SetOrder order;
Mesh.MshFileVersion = 2.2;

Save "mesh.msh";
