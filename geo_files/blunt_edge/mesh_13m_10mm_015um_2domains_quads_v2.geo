clF = 0.001600*1.4142*1.4142;
clB = 0.000800*1.4142;
clM = 0.000800*1.4142;
clS = 0.000600*1.4142;
cl0 = 0.007200*1.4142;
cl1 = 0.024000*1.4142;
cl2 = 0.072000*1.4142;

radius = 1.0;

int_radius = 2.38*25.4/2000;
ext_radius = 2.89*25.4/2000;
burner_base_radius = 4.725*25.4/2000;

burner_height = 0.10;
flame_dist = 0.0015;

mat_location = burner_height + 0.01;  //arbitrary value
mat_wide = 0.5*1.25*25.4/1000;
mat_thick = 0.75*25.4/1000;
insulator = 1.00*25.4/1000 + mat_thick;
insulator_wide = 0.5*1.00*25.4/1000;
mat_holder = 1.75*25.4/1000 + mat_thick;
mat_BL_layer_x = 0.0030;
mat_BL_layer_y = 0.0030;


Point(2) = {ext_radius, burner_height, 0.0, 0.000200};
Point(3) = {int_radius, burner_height, 0.0, 1.0};
Point(4) = {mat_wide + mat_BL_layer_x, burner_height, 0.0, 1.0};
Point(5) = {                       0., burner_height, 0.0, 1.0};

Point(10) = {     0.0, mat_location - mat_BL_layer_y, 0.,     clS};
Point(11) = {     0.0, mat_location                 , 0., 0.7*clS};
Point(12) = {mat_wide, mat_location                 , 0.,     clM};
Point(13) = {mat_wide,      mat_location + mat_thick, 0.,     clM};
Point(14) = {mat_wide,      mat_location + insulator, 0.,     clM};

Point(15) = {mat_wide, mat_location + 1.0*mat_holder, 0., 1.00};
Point(16) = {     0.0, mat_location + 1.0*mat_holder, 0.,  clM};
Point(17) = {     0.0, mat_location + 1.0*mat_holder + mat_BL_layer_y, 0., 1.00*clS};
Point(18) = {     0.0,                          burner_height + 0.550, 0., 6.00*clF};

Point(20) = {              0.00, radius, 0., 12.5*clF};
Point(21) = {              0.25, radius, 0., 12.5*clF};
Point(22) = {            radius, radius, 0.,      cl2};
Point(24) = {            radius,     0., 0.,      cl2};
Point(25) = {burner_base_radius,     0., 0., 6.66*clB};

Point(26) = {burner_base_radius, burner_height-0.0110, 0.0, clB};

Point(28) = {       burner_base_radius, burner_height + 0.000, 0., 1.0};
Point(30) = {               ext_radius, burner_height + flame_dist, 0., 1.0};
Point(32) = {mat_wide + mat_BL_layer_x, burner_height + flame_dist, 0., 1.0};
Point(33) = {                       0., burner_height + flame_dist, 0., clS};

Point(40) = {               ext_radius,                  mat_location - mat_BL_layer_y, 0., 1.0};
Point(41) = {       burner_base_radius,                  mat_location - mat_BL_layer_y, 0., 1.0};
Point(42) = {mat_wide + mat_BL_layer_x,                  mat_location - mat_BL_layer_y, 0., 1.0};
Point(43) = {mat_wide + mat_BL_layer_x,                       mat_location + mat_thick, 0., 1.0};
Point(44) = {mat_wide + mat_BL_layer_x,                       mat_location + insulator, 0., 1.0};
Point(45) = {mat_wide + mat_BL_layer_x, mat_location + 1.0*mat_holder + mat_BL_layer_y, 0., 1.0};
Point(46) = {               ext_radius,                       mat_location + mat_thick, 0., 1.0};
Point(47) = {       burner_base_radius,                       mat_location + mat_thick, 0., 1.0};

Line( 1) = {2, 4};
Line( 2) = {4, 5};
Line( 3) = {4, 32};
Line( 4) = {5, 33};
Line( 5) = {2, 30};
Line( 6) = {30, 32};
Line( 7) = {32, 33};
Line( 8) = {26, 28};
Line( 9) = {26,  2};
Line(10) = {28, 30};

Line(14) = {10, 42};
Line(15) = {33, 10};
Line(16) = {32, 42};
Line(17) = {30, 40};

Line(20) = {41, 47};
Line(21) = {47, 46};
Line(22) = {46, 43};
Line(23) = {28, 41};
Line(24) = {41, 40};
Line(25) = {40, 42};
Line(26) = {40, 46};

Line(30) = {42, 43};
Line(31) = {43, 44};
Line(32) = {44, 45};
Line(33) = {45, 17};

Line(40) = {17, 18};
Line(41) = {18, 20};
Line(42) = {20, 21};
Line(43) = {21, 22};
Line(44) = {22, 24};
Line(45) = {24, 25};
Line(46) = {25, 26};

Line(50) = {10, 11};
Line(51) = {11, 12};
Line(52) = {12, 42};
Line(53) = {13, 43};
Line(54) = {12, 13};
Line(55) = {13, 14};
Line(56) = {14, 15};
Line(57) = {15, 16};
Line(58) = {45, 15};
Line(59) = {16, 17};
Line(60) = {44, 14};

Point(51) = {insulator_wide, mat_location + insulator, 0., clM};
Point(52) = {           0.0, mat_location + insulator, 0., clM};
Point(53) = {           0.0, mat_location + mat_thick, 0., clM};
Point(54) = {insulator_wide, mat_location + mat_thick, 0., clM};
Point(55) = {insulator_wide, mat_location + 0.003, 0., clM};
Point(56) = {           0.0, mat_location + 0.003, 0., clM};

/*Line(70) = {54, 55};*/
/*Line(71) = {55, 12};*/
/*Line(72) = {55, 56};*/
/*Line(73) = {11, 56};*/
/*Line(74) = {56, 53};*/

Line(63) = {11, 53};
Line(64) = {54, 13};
Line(65) = {16, 52};

Line(66) = {51, 52};
Line(67) = {52, 53};
Line(68) = {53, 54};
Line(69) = {54, 51};

/*+++++++++++++++++++++++++++++++++++++++++++++*/

/*horizontal*/
Transfinite Line {  1} = 41 Using Progression 1.015;
Transfinite Line {  6} = 41 Using Progression 1.015;
Transfinite Line { 25} = 41 Using Progression 1.015;
Transfinite Line { 22} = 41 Using Progression 1.015;

Transfinite Line {- 2} = 31 Using Progression 1.0;
Transfinite Line {- 7} = 31 Using Progression 1.0;
Transfinite Line { 14} = 31 Using Progression 1.0;
Transfinite Line {-51} = 31 Using Progression 1.01;

/*vertical*/
Transfinite Line { 15} = 16 Using Progression 1.03;
Transfinite Line { 16} = 16 Using Progression 1.03;
Transfinite Line { 17} = 16 Using Progression 1.03;

Transfinite Line { 23} = 16 Using Progression 1.0;
Transfinite Line {  8} = 21 Using Progression 1.0;

Transfinite Line {  3} = 21 Using Progression 1.15;
Transfinite Line {  5} = 21 Using Progression 1.15;
Transfinite Line { 13} = 21 Using Progression 1.0;
Transfinite Line {  3} = 21 Using Progression 1.15;
Transfinite Line {  4} = 21 Using Progression 1.15;

/*burner*/
Transfinite Line { -9} = 31 Using Progression 1.05;
Transfinite Line {-10} = 31 Using Progression 1.05;
Transfinite Line {-24} = 31 Using Progression 1.05;
Transfinite Line {-21} = 31 Using Progression 1.05;

/*material*/
Transfinite Line {-50} = 13 Using Progression 1.10;
Transfinite Line { 52} = 13 Using Progression 1.10;
Transfinite Line { 53} = 13 Using Progression 1.10;
Transfinite Line {-58} = 13 Using Progression 1.10;
Transfinite Line { 59} = 13 Using Progression 1.10;
Transfinite Line {-60} = 13 Using Progression 1.10;

Transfinite Line { 20} = 31 Using Progression 1.018;
Transfinite Line { 26} = 31 Using Progression 1.018;
Transfinite Line { 30} = 31 Using Progression 1.018;
Transfinite Line { 54} = 31 Using Progression 1.03;

Transfinite Line { 31} = 21 Using Progression 1.02;
Transfinite Line { 55} = 21 Using Progression 1.02;

Transfinite Line {-32} = 16 Using Progression 1.005;
Transfinite Line {-56} = 16 Using Progression 1.02;

Transfinite Line {-33} = 19 Using Progression 1.02;
Transfinite Line { 57} = 19 Using Progression 1.0;

Transfinite Line {-64} = 6 Using Progression 1.0;


Line Loop(41) = {8,23,20,21,22,31:33,40:46};
Plane Surface(1) = {41};

Point(70) = {  0.070, burner_height - 0.028, 0., 2.250*clF};
Point(71) = {  0.110, burner_height - 0.025, 0., 2.500*clF};
Point(72) = {  0.150, burner_height - 0.020, 0., 2.625*clF};
Point(73) = {  0.190, burner_height - 0.012, 0., 2.750*clF};
Point(74) = {  0.220, burner_height + 0.000, 0., 2.875*clF};
Point(75) = {  0.235, burner_height + 0.020, 0., 3.000*clF};
Point(76) = {  0.240, burner_height + 0.050, 0., 3.125*clF};
Point(77) = {  0.240, burner_height + 0.080, 0., 3.250*clF};
Point(78) = {  0.240, burner_height + 0.120, 0., 3.375*clF};
Point(79) = {  0.240, burner_height + 0.160, 0., 3.500*clF};
Point(80) = {  0.240, burner_height + 0.200, 0., 3.625*clF};
Point(81) = {  0.240, burner_height + 0.240, 0., 3.750*clF};
Point(82) = {  0.240, burner_height + 0.280, 0., 4.000*clF};
Point(83) = {  0.240, burner_height + 0.320, 0., 4.250*clF};
Point(84) = {  0.240, burner_height + 0.360, 0., 4.500*clF};
Point(85) = {  0.240, burner_height + 0.400, 0., 4.750*clF};
Point(86) = {  0.240, burner_height + 0.450, 0., 5.000*clF};
Point(87) = {  0.240, burner_height + 0.500, 0., 5.500*clF};
Point(88) = {  0.240, burner_height + 0.550, 0., 6.000*clF};
Point{70:88} In Surface{1};

/*Point(101) = {             0.000500, flame_dist + burner_height, 0., 1.0};*/
/*Point(102) = {int_radius + 0.000281,              burner_height, 0., 1.0};*/
/*Point(103) = {int_radius - 0.000281,              burner_height, 0., 1.0};*/
/*Point(104) = {int_radius           ,   0.000015 + burner_height, 0., 1.0};*/
/*Point(105) = {      0.005 - 0.00050, flame_dist + burner_height, 0., 1.0};*/
/*Point(106) = {      0.005 + 0.00050, flame_dist + burner_height, 0., 1.0};*/
/*Point(107) = {        0.           ,  -0.000120 + mat_location , 0., 1.0};*/

/*Inlet 1*/
Line Loop(42) = {1,3,-6,-5};
Plane Surface(2) = {42};
Transfinite Surface {2};

/*Above inlet 1*/
Line Loop(43) = {17,25,-16,-6};
Plane Surface(3) = {-43};
Transfinite Surface {3};

/*Inlet 2*/
Line Loop(44) = {-3,2,4,-7};
Plane Surface(4) = {44};
Transfinite Surface {4};

/*Above inlet 2*/
Line Loop(45) = {7,15,14,-16};
Plane Surface(5) = {45};
Transfinite Surface {5};

/*Burner tilted edge*/
Line Loop(46) = {-8,9,5,-10};
Plane Surface(6) = {46};
Transfinite Surface {6};

/*Above tilted edge*/
Line Loop(47) = {17,10,-23,-24};
Plane Surface(7) = {47};
Transfinite Surface {7};

/*Above above inlet 2*/
Line Loop(48) = {-26,-22,30,25};
Plane Surface(8) = {48};
Transfinite Surface {8};

/*Above tilted edge*/
Line Loop(49) = {-26,20,21,-24};
Plane Surface(9) = {-49};
Transfinite Surface {9};



/*Material BL*/
Line Loop(63) = {51,52,-14,50};
Plane Surface(11) = {63};
Transfinite Surface {11};

/*Material BL*/
Line Loop(64) = {-53, -54, 52, 30};
Plane Surface(12) = {-64};
Transfinite Surface {12};

/*Material BL*/
Line Loop(65) = {-60,-31, -53, 55};
Plane Surface(13) = {65};
Transfinite Surface {13};

/*Material BL*/
Line Loop(66) = {60, 56, -58, -32};
Plane Surface(14) = {66};
Transfinite Surface {14} Right;

/*Material BL*/
Line Loop(67) = {59,-33, 58, 57};
Plane Surface(15) = {67};
Transfinite Surface {15};


/*Alumina*/
Line Loop(70) = {66,67,68,69};
Plane Surface(16) = {-70};

/*Graphite*/
Line Loop(71) = {55,56,57,65,-66,-69,64};
Plane Surface(17) = {-71};

/*Sample*/
Line Loop(72) = {-54,-51,63,68,64};
Plane Surface(18) = {72};
/*Line Loop(72) = {-54,64,-70,-71};*/
/*Plane Surface(18) = {72};*/
/*Line Loop(73) = {71,-51,73,-72};*/
/*Plane Surface(19) = {73};*/
/*Line Loop(74) = {70,72,74,68};*/
/*Plane Surface(20) = {74};*/

/*#########################*/

Physical Line("inlet") = {1,2,5};
Physical Line("symmetry") = {6,20:22,50,31,32,59};
Physical Line("burner") = {44,14,15};
Physical Line("outlet") = {40};
Physical Line("linear") = {41,42,43};
Physical Line("wall_sym") = {63,65,67};
Physical Surface("fluid") = {1:9,11:15};
Physical Surface("wall_sample") = {18};
Physical Surface("wall_alumina") = {16};
Physical Surface("wall_graphite") = {17};

